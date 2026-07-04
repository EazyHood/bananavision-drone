#!/usr/bin/env python3
"""ROS2 starter node for BananaVision.

This file is intentionally dependency-light and meant as an integration
template. Install ROS2, cv_bridge, and sensor message packages on the target.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image

from bananavision.pipeline import load_config, make_detector, predict_image


def main() -> None:
    try:
        import rclpy
        from cv_bridge import CvBridge
        from rclpy.node import Node
        from sensor_msgs.msg import Image as RosImage
        from std_msgs.msg import String
    except Exception as exc:
        raise RuntimeError("Install ROS2 Python packages and cv_bridge on the drone computer.") from exc

    class BananaCounterNode(Node):
        def __init__(self) -> None:
            super().__init__("banana_counter")
            self.declare_parameter("config", "configs/banana_uav.yaml")
            self.config_path = self.get_parameter("config").get_parameter_value().string_value
            self.config = load_config(self.config_path)
            self.detector = make_detector(self.config)
            self.bridge = CvBridge()
            self.publisher = self.create_publisher(String, "bananavision/detections", 10)
            self.subscription = self.create_subscription(RosImage, "camera/image", self.on_image, 10)

        def on_image(self, msg) -> None:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            pil_image = Image.fromarray(cv_image)
            with NamedTemporaryFile(suffix=".jpg", delete=False) as handle:
                temp_path = Path(handle.name)
            pil_image.save(temp_path)
            try:
                result = predict_image(temp_path, self.config, detector=self.detector)
                payload = String()
                payload.data = json.dumps(result.to_dict())
                self.publisher.publish(payload)
            finally:
                temp_path.unlink(missing_ok=True)

    rclpy.init()
    node = BananaCounterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
