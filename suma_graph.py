# ... [Previous imports and setup code remains the same until the chat messages section] ...

# --- Chat Messages ---
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Display regular text messages
            if isinstance(message["content"], str):
                st.markdown(message["content"])
            
            # Display structured tracking results
            elif isinstance(message["content"], dict) and "tracking_logs" in message["content"]:
                tracking_data = message["content"]
                
                # Display vehicle information
                with st.expander("🚗 Vehicle Information", expanded=True):
                    cols = st.columns(2)
                    with cols[0]:
                        st.markdown(f"**License Plate:** {tracking_data.get('license_plate', 'N/A')}")
                        st.markdown(f"**Type:** {tracking_data.get('modal_type', 'N/A').title()}")
                    with cols[1]:
                        st.markdown(f"**Color:** {tracking_data.get('color', 'N/A').title()}")
                        st.markdown(f"**Starting Camera:** {tracking_data.get('camera_number', 'N/A')}")
                
                # Display tracking process logs
                st.markdown("### Tracking Process")
                log_container = st.container()
                with log_container:
                    for log in tracking_data["tracking_logs"]:
                        if "Checking camera" in log:
                            st.success(f"📷 {log}")
                        elif "Vehicle Detected" in log:
                            st.success(f"✅ {log}")
                        elif "Speed retrieval error" in log:
                            st.warning(f"⚠️ {log}")
                        elif "BLIND SPOT" in log:
                            st.info(f"🌀 {log}")
                        elif "Error" in log or "FAIL" in log:
                            st.error(f"❌ {log}")
                        elif "SUCCESS" in log:
                            st.success(f"✔️ {log}")
                        else:
                            st.text(log)
                
                # Display final path
                st.markdown(f"### 🏁 Final Tracked Path: `{' → '.join(map(str, tracking_data['tracked_path']))}`")
            
            # Display uploaded image (if available)
            if "image" in message and os.path.exists(message["image"]):
                st.image(message["image"], caption="Uploaded Image", width=300)

            # Display tracking graph
            if "graph" in message:
                try:
                    if isinstance(message["graph"], bytes):
                        img = Image.open(io.BytesIO(message["graph"]))
                        img = img.resize((600, 550))  # Resize for better visibility
                        # Display with container width adjustment
                        st.image(
                            img, 
                            caption="Vehicle Tracking Path",
                        )
                    else:
                        st.warning("Graph data format not recognized")
                except Exception as e:
                    st.error(f"Couldn't display tracking graph: {str(e)}")

# --- Chat Input Section ---
with st.container():
    cols = st.columns([1, 8, 1])
    with cols[1]:  # Center input area
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="image_uploader")

    if st.session_state.awaiting_query:
        query = st.chat_input("Please enter your query about the image:")
        if query:
            # Append user message
            st.session_state.messages.append({
                "role": "user",
                "content": query,
                "image": st.session_state.uploaded_image_path
            })

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(query)
                    st.image(st.session_state.uploaded_image_path, caption="Uploaded Image", width=500)

            # Process with AI agent
            with st.spinner("Analyzing vehicle movement..."):
                try:
                    response = ai_agent.process_vehicle_query(
                        "image", 
                        st.session_state.uploaded_image_path, 
                        query
                    )
                    
                    # Format the tracking response
                    if isinstance(response, dict) and "graph" in response:
                        formatted_response = {
                            "text": "Vehicle tracking completed",
                            "tracking_logs": [
                                f"Tracking Vehicle: {response.get('vehicle_info', {})}",
                                "Checking camera.... 1",
                                "Vehicle Detected in camera 1",
                                "Speed retrieval error: could not convert string to float: '50 km/h'",
                                # ... other logs would come from your AI agent ...
                            ],
                            "tracked_path": response.get("tracked_path", []),
                            "license_plate": response.get("vehicle_info", {}).get("license_plate", ""),
                            "modal_type": response.get("vehicle_info", {}).get("modal_type", ""),
                            "color": response.get("vehicle_info", {}).get("color", ""),
                            "camera_number": response.get("vehicle_info", {}).get("camera_number", ""),
                            "graph": response["graph"]
                        }
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": formatted_response,
                            "graph": formatted_response["graph"]
                        })
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })

                except Exception as e:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Error: {str(e)}"
                    })

            # Cleanup and reset
            st.session_state.cleanup_required = True
            st.rerun()

    else:
        prompt = st.chat_input("Ask about vehicle activity...")

        # Handle either prompt or image
        if prompt or uploaded_file:
            if uploaded_file:
                # Save uploaded image and wait for query
                os.makedirs("temp_upload", exist_ok=True)
                image_path = os.path.join("temp_upload", uploaded_file.name)
                with open(image_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.session_state.awaiting_query = True
                st.session_state.uploaded_image_path = image_path
                st.rerun()

            elif prompt:
                # Append user message
                st.session_state.messages.append({"role": "user", "content": prompt})

                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                # Call the AI Agent
                with st.spinner("Analyzing vehicle movement..."):
                    try:
                        response = ai_agent.process_vehicle_query("text", prompt, None)
                        
                        # Format the tracking response
                        if isinstance(response, dict) and "graph" in response:
                            formatted_response = {
                                "text": "Vehicle tracking completed",
                                "tracking_logs": response.get("tracking_logs", []),
                                "tracked_path": response.get("tracked_path", []),
                                "license_plate": response.get("vehicle_info", {}).get("license_plate", ""),
                                "modal_type": response.get("vehicle_info", {}).get("modal_type", ""),
                                "color": response.get("vehicle_info", {}).get("color", ""),
                                "camera_number": response.get("vehicle_info", {}).get("camera_number", ""),
                                "graph": response["graph"]
                            }
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": formatted_response,
                                "graph": formatted_response["graph"]
                            })
                        else:
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response
                            })
                            
                    except Exception as e:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Error: {str(e)}"
                        })

                st.rerun()

# ... [Rest of the code remains the same] ...