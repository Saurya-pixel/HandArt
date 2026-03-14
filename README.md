# HandArt - Hand Motion Drawing

- **Real-time Hand Tracking**: Uses MediaPipe for accurate hand detection and tracking
- **Colorful Trails**: Draw with rainbow colors that cycle automatically
- **Fading Effect**: Trails gradually fade away after 3 seconds of no motion
- **Multi-hand Support**: Track up to 2 hands simultaneously
- **Hand Isolation Mode**: Option to show only your hand with background removed
- **Interactive Controls**: Toggle drawing, clear canvas, and more
- **Laser Sound Effects**: Sci-fi audio feedback as you draw

## Demo

Point with your index finger to draw colorful trails in the air. The trails will automatically fade after a few seconds, creating a dynamic, ephemeral art experience.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Saurya-pixel/HandArt.git
cd HandArt
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.12 or higher
- Webcam
- Dependencies:
  - opencv-python
  - mediapipe>=0.10.0
  - numpy
  - pygame
  - scipy

## Usage

Run the application:
```bash
python hand_drawing.py
```

### Controls

- **Point with your index finger** to draw
- **SPACE** - Toggle drawing on/off
- **E** - Toggle eraser mode
- **H** - Toggle hand isolation mode (show only hand, hide background)
- **C** - Clear all trails from the canvas
- **Q** - Quit the application

## How It Works

1. The application captures video from your webcam
2. MediaPipe detects and tracks your hand(s) in real-time
3. The position of your index finger tip is used as the drawing point
4. As you move your hand, colorful trails are created with glow effects
5. Trails automatically fade out over 3 seconds
6. Each new trail segment gets a different rainbow color
7. **Hand Isolation Mode**: Press 'H' to enable background removal - only your hand will be visible on screen using advanced masking techniques
8. **Sound Effects**: Laser sounds play as you draw with movement-based triggering

## Customization

You can customize the application by modifying parameters in `hand_drawing.py`:

```python
app = HandDrawing(
    fade_duration=3.0,      # Time in seconds before trails fade
    trail_thickness=10      # Thickness of the drawing trails
)
```

## Technical Details

- **Hand Tracking**: Uses MediaPipe Tasks API for robust hand landmark detection
- **Fading Algorithm**: Implements time-based alpha blending for smooth fade effects
- **Performance**: Optimized with deque data structures and efficient point cleanup
- **Color System**: HSV to BGR color conversion for vivid rainbow colors
- **Hand Isolation**: Uses convex hull masking and morphological operations to isolate hand regions from background
- **Audio**: Programmatically generated laser sounds with frequency sweeps and modulation

## Troubleshooting

**Webcam not detected:**
- Make sure your webcam is connected and not being used by another application
- Try changing the camera index in the code: `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)`

**Hand tracking not working:**
- Ensure good lighting conditions
- Keep your hand within the camera frame
- Try adjusting the detection confidence parameters

**Performance issues:**
- Reduce the `trail_thickness` value
- Decrease the `fade_duration`
- Lower your webcam resolution

