auto& preprocessing = graph.AddNode("mediapipe.tasks.components.processors.ImagePreprocessingGraph");
image_in >> preprocessing.In("IMAGE");

auto& hand_detector = graph.AddNode("mediapipe.tasks.vision.hand_detector.HandDetectorGraph");
hand_detector.GetOptions<HandDetectorGraphOptions>().CopyFrom(tasks_options.hand_detector_graph_options());
image_in >> hand_detector.In("IMAGE");
auto hand_rects_from_hand_detector = hand_detector.Out("HAND_RECTS");

auto& hand_landmarks_detector_graph = graph.AddNode("mediapipe.tasks.vision.hand_landmarker.MultipleHandLandmarksDetectorGraph");
hand_landmarks_detector_graph.GetOptions<HandLandmarksDetectorGraphOptions>().CopyFrom(tasks_options.hand_landmarks_detector_graph_options());
image_in >> hand_landmarks_detector_graph.In("IMAGE");
clipped_hand_rects >> hand_landmarks_detector_graph.In("HAND_RECT");
auto landmarks = hand_landmarks_detector_graph.Out("LANDMARKS");

auto& hand_roi_refinement = graph.AddNode("HandRoiRefinementGraph");
auto roi_from_pose = GetHandRoiFromPosePalmLandmarks(pose_palm_landmarks, image_size, graph);
auto roi_from_recrop = RefineHandRoi(image, roi_from_pose, hand_roi_refinement_graph_options, graph);
auto tracking_roi = TrackHandRoi(prev_landmarks, roi_from_recrop, image_size, graph);

hand_presence = IsOverThreshold(score, /*threshold=*/0.1, graph);
auto& landmark_projection = graph.AddNode("LandmarkProjectionCalculator");
landmark_projection.Out("ProjectedLandmarks") >> alignment.In("LANDMARKS");

hand_presence = IsOverThreshold(score, /*threshold=*/0.1, graph);
auto& render_landmarks = graph.AddNode("RenderLandmarksCalculator");
landmarks >> render_landmarks.In("LANDMARKS");
render_landmarks.Out("RENDERED_IMAGE") >> pass_through.In("IMAGE");

return {
    /* landmark_lists= */ filtered_landmarks,
    /* world_landmark_lists= */ filtered_world_landmarks,
    /* hand_rects_next_frame= */ filtered_hand_rects_for_next_frame,
    /* handedness= */ filtered_handedness,
    /* palm_rects= */ hand_detector.Out("PALM_RECTS"),
    /* palm_detections= */ hand_detector.Out("PALM_DETECTIONS"),
    /* image= */ pass_through.Out("IMAGE")
};