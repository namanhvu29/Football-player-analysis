# Football Player Analysis System

## Overview
This project is an advanced computer vision system designed to analyze football game footage. It utilizes state-of-the-art machine learning models to detect and track players, referees, and the ball. The system analyzes player movements, assigns teams based on jersey colors, estimates camera movement to stabilize tracking, and calculates player speed and distance covered.

## Features

- **Object Detection & Tracking**: Utilizes YOLO (You Only Look Once) for detecting players, referees, and the ball, combined with ByteTrack for robust object tracking across frames.
- **Team Assignment**: Automatically assigns players to teams based on the color of their jerseys using K-Means clustering.
- **Camera Movement Estimation**: Estimates and compensates for camera movement to provide accurate world-coordinate positions for players.
- **Perspective Transformation**: Transforms the 2D video perspective into a top-down view for accurate distance measurements.
- **Speed & Distance Estimation**: Calculates the speed (km/h) and total distance (m) covered by each player throughout the video.
- **Ball Control Analysis**: detailed analysis of ball possession by team.

## Prerequisites

- **Python 3.10+**
- **Docker** (Optional, for containerized execution)

## Installation & Usage

### Option 1: Running with Docker (Recommended)

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Football-player-analysis
    ```

2.  **Build and Run:**
    ```bash
    docker-compose up --build
    ```
    This command will build the Docker image and start the processing.
    - Input videos are read from the `./input_videos` directory.
    - Processed videos are saved to the `./output_videos` directory.

3.  **Stop:**
    ```bash
    docker-compose down
    ```

### Option 2: Running Locally

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Football-player-analysis
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Analysis:**
    ```bash
    python main.py
    ```

## Directory Structure

- `camera_movement_estimator/`: Logic for estimating camera motion.
- `input_videos/`: Directory to place input video files.
- `models/`: Directory containing YOLO model weights (`best.pt`).
- `output_videos/`: Directory where processed videos are saved.
- `player_ball_assigner/`: Logic to determine which player has the ball.
- `speed_and_distance_estimator/`: Logic for calculating player statistics.
- `stubs/`: Cache files for intermediate results (e.g., detections) to speed up development.
- `team_assigment/`: Logic for clustering player colors and assigning teams.
- `trackers/`: Object tracking implementation.
- `utils/`: Helper functions for video I/O and bounding box operations.
- `view_transformer/`: Logic for perspective transformation.
- `main.py`: The entry point script for the application.
- `Dockerfile` & `docker-compose.yml`: Docker configuration files.

## Credits
This project integrates various computer vision techniques to provide a comprehensive analysis tool for football analytics.