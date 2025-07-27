import io
import requests
import json
import os
import sys
import autogen
import cv2
import numpy as np
import re
from pymongo import MongoClient
from datetime import datetime
from sentence_transformers import SentenceTransformer
from suma_yolo import process_video_with_yolo
from graph1 import get_connected_nodes, get_direction  # Fixed single graph import
from graph import check_camera
from suma_image import process_image
import networkx as nx
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from route_predictor import predict_next_node

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# --- Configuration Constants ---
TUNED_MODEL_NAME = "tunedModels/unisys-maikwyl6uzmz"
API_KEY = "AIzaSyANV5EmDmp09BXLxl5gv3_RBGT2uaqGeTo"
API_URL = f"https://generativelanguage.googleapis.com/v1beta/{TUNED_MODEL_NAME}:generateContent?key={API_KEY}"
BLIND_SPOTS = {7}
MONGO_URI = "mongodb+srv://AI_agent:z8W1L0n41kZvseDw@unisys.t75li.mongodb.net/?retryWrites=true&w=majority&appName=Unisys"
VIDEO_BASE_DIR = "E:/Unisys2new/DEMO_New" if os.name == "nt" else "/home/user/Unisys2new/DEMO_New"

# --- MongoDB Setup ---
try:
    client = MongoClient(MONGO_URI)
    db = client["vehicle_detection"]
    collection = db["detected_vehicles"]
except Exception as e:
    print(f"MongoDB connection error: {e}")
    exit()

# --- Helper Classes ---
class VehicleState:
    """Tracks vehicle movement state with initial state handling"""
    def __init__(self, vehicle_id):
        self.vehicle_id = vehicle_id
        self.previous_node = None
        self.current_node = None
        self.speed = None
        self.direction = "East"
    # def initialize(self, start_node):
    #     """Call this for initial state setup"""
    #     self.previous_node = start_node
    #     self.current_node = start_node
    #     self.direction = get_direction(start_node, start_node)
    def update(self, new_node, speed):
        # Handle initial state
        if self.current_node is None:
            self.previous_node = new_node
            self.current_node = new_node
            self.direction = get_direction(new_node, new_node)
        else:
            self.previous_node = self.current_node
            self.current_node = new_node
            self.direction = get_direction(self.previous_node, self.current_node)
        self.speed = speed

        # Fallback direction validation
        valid_directions = ['East', 'West', 'South', 'North', 'Initial-East']
        if self.direction not in valid_directions:
            self.direction = "East"
        
        # Handle initial state direction
        if self.previous_node is None:
            self.direction = "East"
        else:
            self.direction = get_direction(self.previous_node, self.current_node)
# --- Core Functions ---
def handle_blind_spot(state, predicted_node):
    """Handles nodes without video feeds using graph connections"""
    connected = get_connected_nodes(predicted_node)
    for node in connected:
        temp_state = VehicleState(state.vehicle_id)
        temp_state.previous_node = state.current_node
        temp_state.current_node = predicted_node
        temp_state.speed = state.speed
        temp_state.direction = get_direction(predicted_node, node)
        
        next_node = predict_next_node(
            temp_state.previous_node,
            temp_state.current_node,
            temp_state.speed,
            temp_state.direction
        )
        if next_node not in BLIND_SPOTS:
            return next_node
    return None

def fuzzy_match_plates(original, detected, threshold=65):
    """Improved fuzzy matching with normalization"""
    clean = lambda s: re.sub(r'[^A-Z0-9]', '', s.upper())
    return fuzz.ratio(clean(original), clean(detected)) > threshold
def clean_llm_response(response_text):
    """
    Cleans the LLM response by removing any corrupted patterns and extracting valid JSON.
    Handles cases like:
    - Proper JSON
    - JSON with corrupted prefixes/suffixes
    - JSON with repeated pattern corruption
    """
    if not response_text:
        return None
    
    # Debug: Print raw response for troubleshooting
    # print("Raw LLM Response:", response_text[:500] + "..." if len(response_text) > 500 else response_text)
    
    # Case 1: Try to parse directly as JSON first
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass
    
    # Case 2: Clean common corruption patterns
    cleaned_response = re.sub(r'\{\{+', '{', response_text)  # Remove duplicate {
    cleaned_response = re.sub(r'\}\}+', '}', cleaned_response)  # Remove duplicate }
    cleaned_response = re.sub(r'\(\)+', '', cleaned_response)  # Remove () patterns
    
    # Case 3: Try to extract JSON from within the text
    try:
        # Look for the outermost complete JSON object
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except:
        pass
    
    # Case 4: Fallback - try to build JSON from known fields
    try:
        result = {}
        if 'number_plate' in response_text:
            plate_match = re.search(r'"number_plate":\s*"([^"]+)"', response_text)
            if plate_match:
                result['number_plate'] = plate_match.group(1)
        
        if 'vehicle_name' in response_text:
            vehicle_match = re.search(r'"vehicle_name":\s*"([^"]+)"', response_text)
            if vehicle_match:
                result['vehicle_name'] = vehicle_match.group(1)
        
        if 'color' in response_text:
            color_match = re.search(r'"color":\s*"([^"]+)"', response_text)
            if color_match:
                result['color'] = color_match.group(1)
        
        if 'camera' in response_text:
            camera_match = re.search(r'"camera":\s*(\d+)', response_text)
            if camera_match:
                result['camera'] = int(camera_match.group(1))
        
        if result:
            result['track'] = True  # Default to tracking if we found any data
            return result
    except:
        pass
    
    return None

def call_ai_studio(prompt_text):
    """Send prompt to AI Studio and return the structured response"""
    # Very explicit prompt to get clean JSON
    structured_prompt = f"""
    Extract vehicle tracking information and return ONLY a valid JSON object with these exact fields:
    {{
        "number_plate": "string (vehicle license plate, empty if unknown)",
        "vehicle_name": "string (vehicle type like sedan, minivan)",
        "company": "string (manufacturer if known, empty if unknown)",
        "color": "string",
        "camera": number,
        "time": "string (optional time if mentioned)",
        "track": boolean
    }}

    Example Response:
    {{
        "number_plate": "PY05W2725",
        "vehicle_name": "minivan",
        "company": "",
        "color": "gray",
        "camera": 1,
        "track": true
    }}

    Input query: {prompt_text}
    """
    
    headers = {'Content-Type': 'application/json'}
    data = {"contents": [{"parts": [{"text": structured_prompt}]}]}
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        response_data = response.json()
        
        try:
            generated_text = response_data['candidates'][0]['content']['parts'][0]['text']
            return clean_llm_response(generated_text)
        except (KeyError, IndexError) as e:
            print("Error parsing API response structure")
            return None
            
    except Exception as e:
        print(f"API call failed: {str(e)}")
        return None

def parse_llm_response(response_text):
    """
    Parse the LLM's text response into a structured dictionary.
    Example response format we expect:
    {
        "number_plate": "TN01Ae3737",
        "vehicle_name": "sedan",
        "company": "Toyota",
        "color": "dark blue",
        "camera": 71,
        "time": "15:00",
        "track": true
    }
    """
    try:
        # The response should be in JSON format
        return json.loads(response_text)
    except json.JSONDecodeError:
        print("Failed to parse LLM response as JSON")
        return None
# --- AI Agent Class ---
class AIAgent(autogen.AssistantAgent):
    def __init__(self, name="VehicleTrackingAgent"):
        super().__init__(name=name)
        self.vehicle_states = {}
        self.visited_cameras = set()
        self.collection = db["detected_vehicles"]
        # self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_multimodal_input(self, input_type, user_input, user_query=None):
        """Handles input by either processing directly or via AI Studio LLM"""
        if input_type == "text":
            # Send the text query to AI Studio first
            llm_response = call_ai_studio(user_input)
            if not llm_response:
                print("No valid data from AI Studio, falling back to direct processing")
                # llm_response = get_multimodal_data(query=user_input)
        elif input_type == "image":
            if not os.path.exists(user_input):
                print(f"Image file not found: {user_input}")
                return None
            if not user_query:
                print("No query provided for image processing")
                return None
            llm_response = process_image(user_input, user_query)
        
        if not llm_response:
            print("No valid data extracted from input")
            return None
            
        return self._normalize_data(llm_response)

    def _normalize_data(self, raw_data):
        """Convert multimodal output to standardized format with proper defaults."""
        if not raw_data or not isinstance(raw_data, dict):
            print("Invalid raw data received for normalization")
            return None

        # Ensure all required fields have default values
        return {
            "license_plate": str(raw_data.get("number_plate", raw_data.get("license_plate", ""))).upper(),
            "modal_type": str(raw_data.get("vehicle_name", raw_data.get("modal_type", ""))).lower(),
            "company": str(raw_data.get("company", "")).lower(),
            "color": str(raw_data.get("color", "")).lower(),
            "camera_number": int(raw_data.get("camera", 1)),
            "should_track": bool(raw_data.get("track", False))
        }

    def process_vehicle_query(self, input_type, user_input, user_query=None):
        """Main function that handles both tracking and specific camera search."""
        multimodal_data = self.get_multimodal_input(input_type, user_input, user_query)
        if not multimodal_data:
            return "No valid multimodal data received."

        # Check if we should track or search specific camera
        if multimodal_data.get("should_track"):
            return self.track_vehicle(input_type, user_input, user_query)
        elif multimodal_data.get("camera_number"):
            return self.search_in_specific_camera(
                input_type, 
                user_input, 
                multimodal_data["camera_number"],
                user_query
            )
        else:
            # Default behavior if no specific instruction
            return self.track_vehicle(input_type, user_input, user_query)

    def search_in_specific_camera(self, input_type, user_input, camera_number, user_query=None):
        """Searches for a vehicle only in the specified camera without tracking across cameras."""
        multimodal_data = self.get_multimodal_input(input_type, user_input, user_query)
        if not multimodal_data:
            return "No valid multimodal data received."

        # Validate identification parameters
        if not any([multimodal_data["license_plate"], 
                    multimodal_data["modal_type"],
                    multimodal_data["company"], 
                    multimodal_data["color"]]):
            return "Insufficient vehicle identification parameters provided."

        vehicle_info = {
            "license_plate": multimodal_data["license_plate"].upper(),
            "modal_type": multimodal_data["modal_type"].lower(),
            "company": multimodal_data["company"].lower(),
            "color": multimodal_data["color"].lower(),
            "camera_number": int(camera_number)
        }

        if not check_camera(vehicle_info["camera_number"]):
            return f"Error: Camera {vehicle_info['camera_number']} does not exist."

        print(f"Searching for Vehicle in Camera {vehicle_info['camera_number']}: {vehicle_info}")

        # Process only the specified camera
        if self.process_camera_feed(vehicle_info["camera_number"], vehicle_info):
            return f"Vehicle found in camera {vehicle_info['camera_number']}"
        return f"Vehicle not detected in camera {vehicle_info['camera_number']}"
    # --- Core Tracking Methods ---
    def track_vehicle(self, input_type, user_input, user_query=None):
        """Predictive tracking with visualization"""
        multimodal_data = self.get_multimodal_input(input_type, user_input, user_query)
        if not multimodal_data:
            return {"text": "Error: Could not process input data."}

        # Structured vehicle information
        vehicle_info = {
            "license_plate": multimodal_data.get("license_plate", "").upper(),
            "modal_type": multimodal_data.get("modal_type", "").lower(),
            "company": multimodal_data.get("company", "").lower(),
            "color": multimodal_data.get("color", "").lower(),
            "camera_number": multimodal_data.get("camera_number", 1)
        }

        # Validate identification parameters
        if not any([vehicle_info["license_plate"], 
                    vehicle_info["modal_type"],
                    vehicle_info["company"], 
                    vehicle_info["color"]]):
            return {"text": "Error: Insufficient vehicle identification parameters provided."}

        # Camera validation
        if not check_camera(vehicle_info["camera_number"]):
            return {"text": f"Error: Camera {vehicle_info['camera_number']} does not exist."}
        
        print(f"Tracking Vehicle: {vehicle_info}")

        # Generate vehicle ID
        if vehicle_info["license_plate"]:
            vehicle_id = vehicle_info["license_plate"]
        else:
            identification_string = f"{vehicle_info['modal_type']}_{vehicle_info['company']}_{vehicle_info['color']}"
            vehicle_id = str(hash(identification_string))

        # Initialize tracking state
        state = self.vehicle_states.get(vehicle_id, VehicleState(vehicle_id))
        initial_node = vehicle_info["camera_number"]
        tracked_path = [initial_node]
        max_steps = 10

        # Step 1: Check vehicle in initial camera
        if not self.process_initial_camera(initial_node, multimodal_data):
            return {"text": f"Vehicle not found in initial camera {initial_node}"}

        # Initialize state with validated initial camera
        state.update(initial_node, self._get_speed_from_mongodb(vehicle_id, initial_node))
        


        # Tracking loop
        while max_steps > 0:
            try:
                # Step 2: Predict next node
                predicted = predict_next_node(
                    state.previous_node,
                    state.current_node,
                    state.speed,
                    state.direction
                )
                
                # Validate prediction
                if not predicted or predicted == state.current_node:
                    break
                
                # Step 3: Handle blind spots
                if predicted in BLIND_SPOTS:
                    print(f"[BLIND SPOT] Entering blind spot at node {predicted}")
                    predicted = self.handle_blind_spot_transition(state, predicted)
                    if not predicted:
                        break
                    print(f"[BLIND SPOT] Exited to node {predicted}")

                # Step 4: Verify in next camera (if not blind spot)
                if predicted not in BLIND_SPOTS:
                    if not self.process_camera_feed(predicted, multimodal_data):
                        break  # Vehicle lost
                
                # Step 5: Update state
                new_speed = self._get_speed_from_mongodb(vehicle_id, predicted)
                state.update(predicted, new_speed)
                tracked_path.append(predicted)
                max_steps -= 1

            except Exception as e:
                print(f"Tracking error: {e}")
                break

        # Finalize tracking
        self.vehicle_states[vehicle_id] = state
        
        # Generate the graph visualization
        graph_buffer = self.visualize_tracking_path(tracked_path)
        
        # Return both text response and graph image
        return {
            "text": f"Tracked path: {tracked_path}" if tracked_path else "Tracking lost",
            "graph": graph_buffer.getvalue()  # Return bytes
        }

    def process_initial_camera(self, camera_id, vehicle_info):
        """Mandatory check in user-specified starting camera"""
        print(f"Checking initial camera {camera_id}...")
        return self.process_camera_feed(camera_id, vehicle_info)

    def handle_blind_spot_transition(self, state, predicted_node):
        """Handle blind spot nodes using graph connections"""
        print(f"Handling blind spot at node {predicted_node}")
        connected = get_connected_nodes(predicted_node)
        
        for next_node in connected:
            # Create temp state for prediction
            temp_state = VehicleState(state.vehicle_id)
            temp_state.previous_node = state.current_node
            temp_state.current_node = predicted_node
            temp_state.speed = state.speed
            temp_state.direction = get_direction(predicted_node, next_node)
            
            # Predict through blind spot
            inferred_node = predict_next_node(
                temp_state.previous_node,
                temp_state.current_node,
                temp_state.speed,
                temp_state.direction
            )
            if inferred_node and inferred_node not in BLIND_SPOTS:
                return inferred_node
        return None
    def _get_speed_from_mongodb(self, vehicle_id, node):
        """Retrieves latest speed record from MongoDB"""
        try:
            record = self.collection.find_one(
                {"license_plate": vehicle_id, "camera_number": node},
                sort=[("timestamp", -1)]
            )
            return float(record["speed"]) if record and "speed" in record else 40.0
        except Exception as e:
            print(f"Speed retrieval error: {e}")
            return 40.0

    def visualize_tracking_path(self, tracked_path):
        """Create a visual representation of the tracking path"""
        # Create a directed graph
        G = nx.DiGraph()
        
        # Get all connected nodes from the graph module
        all_edges = []
        for node in tracked_path:
            connected = get_connected_nodes(node)
            for neighbor in connected:
                all_edges.append((node, neighbor))
        
        G.add_edges_from(all_edges)

        # Prepare labels for each node with visit order
        labels = {}
        visit_count = {node: 0 for node in G.nodes}
        order = {}

        # Populate the visit order
        for i, node in enumerate(tracked_path):
            visit_count[node] += 1
            order[node, visit_count[node]] = i + 1

        # Generate node labels including visit order
        for node in G.nodes:
            if visit_count[node] > 0:
                label_list = [str(order[node, count]) for count in range(1, visit_count[node] + 1)]
                labels[node] = f"{node} ({', '.join(label_list)})"
            else:
                labels[node] = str(node)

        # Edge colors and labels
        edge_colors = []
        edge_labels = {}
        visited_edges = list(zip(tracked_path, tracked_path[1:]))

        # Dictionary to keep track of edge traversal counts
        edge_traversal_count = {}

        for u, v in visited_edges:
            if (u, v) not in edge_traversal_count:
                edge_traversal_count[(u, v)] = 1
            else:
                edge_traversal_count[(u, v)] += 1

        for u, v in G.edges:
            if (u, v) in visited_edges:
                edge_colors.append('blue')
                # Count how many times the edge was traversed and label accordingly
                count = edge_traversal_count.get((u, v), 0)
                edge_labels[(u, v)] = f"{u}→{v} ({count}x)" if count > 1 else f"{u}→{v}"
            else:
                edge_colors.append('gray')

        # Draw the graph
        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 8))
        nx.draw(G, pos, 
                with_labels=True, 
                labels=labels, 
                node_color=['lightgreen' if visit_count[node] > 0 else 'lightgray' for node in G.nodes], 
                edge_color=edge_colors, 
                node_size=2000, 
                font_size=12, 
                font_weight='bold')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

        plt.title(f"Vehicle Tracking Path: {' → '.join(map(str, tracked_path))}")
        # plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return buf
    
    def process_camera_feed(self, camera_id, vehicle_info):
        """Processes camera feed using YOLO and checks database matches."""
        file_path = self.get_video_file_path(f"{camera_id}.mp4")
        if not file_path:
            print(f"No video found for camera {camera_id}")
            return False

        try:
            detection_results = process_video_with_yolo(file_path, camera_id)
            return (self.direct_match(detection_results, vehicle_info) or 
                    self.match_vehicle_with_database(vehicle_info, camera_id))
        except Exception as e:
            print(f"Error processing camera {camera_id}: {e}")
            return False

    def direct_match(self, detection_results, vehicle_info):
        """Checks real-time YOLO results for direct match."""
        for result in detection_results:
            if isinstance(result, dict):
                # Use fuzzy matching for license plates
                if vehicle_info["license_plate"] and result.get("license_plate"):
                    if fuzzy_match_plates(vehicle_info["license_plate"], result.get("license_plate", "")):
                        # If license plates match, check other attributes
                        if all([
                            vehicle_info["modal_type"].lower() == result.get("modal_type", "").lower(),
                            vehicle_info["company"].lower() == result.get("company", "").lower(),
                            vehicle_info["color"].lower() == result.get("color", "").lower()
                        ]):
                            return True
        return False

    def match_vehicle_with_database(self, vehicle_info, camera_id):
        """
        Matches vehicle based on priority with fuzzy matching for license plates:
          1. License Plate (fuzzy match)
          2. Last 4 Digits of License Plate (exact match)
          3. Modal Type
          4. Color
        """
        try:
            camera_query = {"camera_number": int(camera_id)}
            camera_records = list(self.collection.find(camera_query))

            if not camera_records:
                print(f"No records found for camera {camera_id}")
                return False

            # License plate fuzzy matching
            license_plate = vehicle_info.get("license_plate", "").strip()
            if license_plate:
                for record in camera_records:
                    record_plate = str(record.get("license_plate", "")).strip()
                    if record_plate and fuzzy_match_plates(license_plate, record_plate):
                        print(f"License plate Detected in camera {camera_id}")
                        return True

                # Check last 4 digits if fuzzy match fails
                if len(license_plate) >= 4:
                    last_four = license_plate[-4:]
                    for record in camera_records:
                        record_plate = str(record.get("license_plate", "")).strip()
                        if len(record_plate) >=4 and record_plate.endswith(last_four):
                            print(f"License plate Detected in camera {camera_id}")
                            return True

            # Other attributes
            priority_order = ["modal_type", "company", "color"]
            for field in priority_order:
                input_value = vehicle_info.get(field, "").strip().lower()
                if not input_value:
                    continue
                
                for record in camera_records:
                    record_value = str(record.get(field, "")).strip().lower()
                    if record_value == input_value:
                        print(f"Match found by {field} in camera {camera_id}")
                        return True

            print(f"No matches found in camera {camera_id}")
            return False
     
        except Exception as e:
            print(f"Database matching error: {e}")
            return False

    def get_video_file_path(self, filename):
        """Returns full path for video files."""
        base_dir = "E:/Unisys3/DEMO_New" if os.name == "nt" else "/home/user/Unisys3/DEMO_New"
        return os.path.join(base_dir, filename) if os.path.exists(os.path.join(base_dir, filename)) else None

    def vector_search(self, query_text, camera_id):
        """Retrieves vehicles using semantic similarity (optional)."""
        try:
            query_embedding = embedding_model.encode(query_text).tolist()
            vehicles = list(collection.find({"camera_number": camera_id}, 
                                         {"embedding": 1, "camera_number": 1, "track_id": 1}))

            return sorted([
                {
                    "track_id": v["track_id"],
                    "camera_number": v["camera_number"],
                    "similarity": np.dot(query_embedding, v["embedding"]) / 
                                (np.linalg.norm(query_embedding) * np.linalg.norm(v["embedding"]))
                } for v in vehicles if "embedding" in v
            ], key=lambda x: x["similarity"], reverse=True)
        except Exception as e:
            print(f"Vector search error: {e}")
            return []
        

if __name__ == "__main__":
    agent = AIAgent()
    input_type = input("Enter input type (text/image): ").strip().lower()
    
    # Ask for different input based on input type
    if input_type == "image":
        user_input = input("Enter the path to the image: ").strip()
        user_query = input("Please enter your query about the image: ").strip()
    elif input_type == "text":
        user_input = input("Enter the description of the vehicle: ").strip()
        user_query = None
    else:
        print("Invalid input type. Please enter either 'text' or 'image'")
        exit()
    
    print("\nFinal Result:", agent.process_vehicle_query(input_type, user_input, user_query))
