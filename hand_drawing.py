#!/usr/bin/env python3
"""
Hand Motion Drawing Application
Tracks hand movements via webcam and draws colorful trails that fade over time.
"""

import cv2
import numpy as np
from scipy import signal
from collections import deque
import time
import pygame
import threading
import os
import wave
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


class HandDrawing:
    """Main application class for hand tracking and drawing."""

    def __init__(self, fade_duration=3.0, trail_thickness=10):
        """
        Initialize the hand drawing application.

        Args:
            fade_duration: Time in seconds before trails completely fade
            trail_thickness: Thickness of the drawing trails
        """
        # Initialize pygame mixer for audio (optional)
        self.mixer_available = False
        try:
            pygame.mixer.init(frequency=22050, channels=1, size=-16)
            self.mixer_available = True
        except (ImportError, NotImplementedError):
            print("⚠ Audio mixer not available - sound effects disabled")

        self.brush_sounds = []
        self.current_sound_index = 0
        self.sound_files = []
        if self.mixer_available:
            self.generate_brush_sounds()
        self.last_sound_time = {}
        self.sound_cooldown = 0.05  # Minimum time between sound plays for continuous effect
        self.last_sound_position = {}  # Track position where sound was last played per hand
        self.movement_threshold = 8  # Pixels of movement required to play sound
        
        # MediaPipe hand tracking setup using new tasks API
        base_options = python.BaseOptions(model_asset_path='/tmp/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        try:
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        except:
            # Fallback: download the model if file doesn't exist
            import urllib.request
            os.makedirs('/tmp', exist_ok=True)
            model_url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
            urllib.request.urlretrieve(model_url, '/tmp/hand_landmarker.task')
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)

        # Drawing parameters
        self.fade_duration = fade_duration
        self.trail_thickness = trail_thickness

        # Canvas and trail storage
        self.canvas = None
        self.trails = []  # List of trails: [(points, timestamp, color)]

        # Color cycling
        self.current_color_index = 0
        self.colors = self.generate_rainbow_colors(6)

        # State
        self.drawing_enabled = True
        self.eraser_enabled = False  # Toggle for eraser mode
        self.hand_isolation_enabled = False  # Toggle to show only hands
        self.last_position = {}  # Track last position per hand

    def generate_rainbow_colors(self, n=6):
        """Generate rainbow colors in BGR format for OpenCV."""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            hsv = np.uint8([[[hue, 255, 255]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            colors.append(tuple(map(int, bgr[0][0])))
        return colors

    def get_next_color(self):
        """Get the next color in the rainbow cycle."""
        color = self.colors[self.current_color_index]
        self.current_color_index = (self.current_color_index + 1) % len(self.colors)
        return color

    def generate_brush_sounds(self):
        """
        Generate smooth laser/sci-fi stroke sounds programmatically.
        Creates electronic-sounding sweeping tones with modulation.
        Also saves them as WAV files for system playback fallback.
        """
        sample_rate = 22050
        duration = 0.15  # 150ms duration
        
        # Create temp directory for sounds if it doesn't exist
        os.makedirs('/tmp/ai_art_sounds', exist_ok=True)
        
        # Create 3 different laser/sci-fi stroke sounds with sweeping frequencies
        for idx, params in enumerate([(400, 800, 0.7), (500, 900, 0.75), (450, 850, 0.65)]):
            start_freq, end_freq, wobble_amount = params
            
            # Create smooth laser-like sound with frequency sweep
            t = np.linspace(0, duration, int(sample_rate * duration))
            
            # Create frequency sweep (upward sweep like a laser)
            freq_sweep = np.linspace(start_freq, end_freq, len(t))
            
            # Generate smooth sine wave with sweeping frequency
            phase = 2 * np.pi * np.cumsum(freq_sweep) / sample_rate
            samples = np.sin(phase)
            
            # Add smooth modulation/wobble for sci-fi effect (not too much)
            wobble_freq = 5 + idx  # Different wobble frequencies
            wobble = np.sin(2 * np.pi * wobble_freq * t) * wobble_amount
            samples = samples * (1 + wobble * 0.15)
            
            # Add slight pulse modulation for electronic feel
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
            samples = samples * pulse
            
            # Apply smooth envelope (quick attack, smooth decay) for laser feel
            envelope = signal.windows.hann(len(samples))
            samples = samples * envelope
            
            # Normalize
            max_val = np.max(np.abs(samples))
            if max_val > 0:
                samples = np.int16(samples * 32767 * 0.8 / max_val)
            else:
                samples = np.int16(samples * 32767 * 0.8)
            
            # Create pygame Sound object (mono audio)
            sound = pygame.sndarray.make_sound(samples)
            self.brush_sounds.append(sound)
            
            # Also save as WAV file for fallback playback
            wav_path = f'/tmp/ai_art_sounds/brush_sound_{idx}.wav'
            with wave.open(wav_path, 'wb') as wav_file:
                wav_file.setnchannels(1)  # mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(samples.tobytes())
            self.sound_files.append(wav_path)
            
            print(f"✓ Laser sound {idx + 1} generated (sweep: {start_freq}→{end_freq}Hz)")

    def play_brush_sound(self, hand_id=0):
        """
        Play a brush stroke sound with sound cooldown to avoid overlapping.
        Tries pygame first, then falls back to system audio command.

        Args:
            hand_id: Identifier for which hand (0 or 1)
        """
        if not self.mixer_available or len(self.brush_sounds) == 0:
            return

        current_time = time.time()

        # Check if enough time has passed since last sound for this hand
        if hand_id in self.last_sound_time:
            if current_time - self.last_sound_time[hand_id] < self.sound_cooldown:
                return

        self.last_sound_time[hand_id] = current_time

        # Play sound in a separate thread to avoid blocking
        def play_sound():
            try:
                sound_idx = self.current_sound_index
                self.current_sound_index = (self.current_sound_index + 1) % len(self.brush_sounds)

                # Try pygame first
                sound = self.brush_sounds[sound_idx]
                sound.set_volume(0.8)
                sound.play()

                # Also try system audio with afplay as backup on macOS
                if len(self.sound_files) > 0:
                    wav_file = self.sound_files[sound_idx]
                    os.system(f'afplay "{wav_file}" 2>/dev/null &')

            except Exception as e:
                # Try fallback system audio
                try:
                    if len(self.sound_files) > 0:
                        wav_file = self.sound_files[sound_idx]
                        os.system(f'afplay "{wav_file}" 2>/dev/null &')
                except:
                    pass

        sound_thread = threading.Thread(target=play_sound, daemon=True)
        sound_thread.start()

    def add_trail_point(self, x, y, hand_id=0, finger_points=None):
        """
        Add a point to the drawing trail and draw to persistent canvas.

        Args:
            x, y: Coordinates of the main point (index finger tip)
            hand_id: Identifier for which hand (0 or 1)
            finger_points: List of additional finger positions [(x1, y1), (x2, y2), ...] for eraser multi-point support
        """
        current_time = time.time()

        # Get or create trail for this hand
        if hand_id not in self.last_position:
            # New trail for this hand
            color = self.get_next_color()
            self.trails.append({
                'points': deque(maxlen=1000),
                'timestamps': deque(maxlen=1000),
                'color': color,
                'hand_id': hand_id
            })
            self.last_position[hand_id] = len(self.trails) - 1
            # Initialize sound position tracking
            self.last_sound_position[hand_id] = (x, y)

        trail_idx = self.last_position[hand_id]
        trail = self.trails[trail_idx]

        # Add point to trail
        trail['points'].append((x, y))
        trail['timestamps'].append(current_time)
        
        # Draw to persistent canvas
        if self.canvas is not None and len(trail['points']) >= 2:
            prev_point = trail['points'][-2]
            curr_point = (x, y)
            
            if self.eraser_enabled:
                # Eraser mode: draw with black (erases the drawing)
                # Use circles at multiple finger points for broad coverage
                eraser_radius = int(self.trail_thickness * 6)
                
                # Erase at main point (index finger)
                cv2.circle(self.canvas, curr_point, eraser_radius, (0, 0, 0), -1)
                
                # Erase at additional finger points if provided (middle and ring fingers)
                if finger_points:
                    for fp in finger_points:
                        cv2.circle(self.canvas, fp, eraser_radius, (0, 0, 0), -1)
            else:
                # Drawing mode: draw with current color and glow effects
                color = trail['color']
                
                # Draw glow layer
                glow_color = tuple(int(c * 0.4) for c in color)
                cv2.line(self.canvas, prev_point, curr_point, glow_color, 
                        int(self.trail_thickness * 2.5))
                
                # Draw main stroke
                cv2.line(self.canvas, prev_point, curr_point, color, self.trail_thickness)
                
                # Draw highlight
                highlight_color = tuple(min(255, int(c * 1.5)) for c in color)
                cv2.line(self.canvas, prev_point, curr_point, highlight_color, 
                        max(1, int(self.trail_thickness * 0.4)))
                
                # Add sparkles occasionally
                if len(trail['points']) % 3 == 0:
                    for _ in range(2):
                        offset_x = np.random.randint(-6, 7)
                        offset_y = np.random.randint(-6, 7)
                        sparkle_pos = (x + offset_x, y + offset_y)
                        cv2.circle(self.canvas, sparkle_pos, 2, highlight_color, -1)
        
        # Only play sound if hand has moved significantly
        if hand_id in self.last_sound_position:
            last_x, last_y = self.last_sound_position[hand_id]
            distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
            
            # Only play sound if movement threshold is exceeded
            if distance >= self.movement_threshold:
                self.last_sound_position[hand_id] = (x, y)
                self.play_brush_sound(hand_id)
        else:
            self.last_sound_position[hand_id] = (x, y)

    def draw_trails(self, frame):
        """
        Composite the persistent canvas onto the frame.
        The canvas contains all permanent strokes drawn so far.

        Args:
            frame: The video frame to draw on
        """
        if self.canvas is not None:
            # Blend canvas with frame (canvas on top)
            mask = np.any(self.canvas != 0, axis=2)
            frame[mask] = self.canvas[mask]

    def clean_old_points(self):
        """Remove points that are older than fade_duration."""
        current_time = time.time()

        for trail in self.trails:
            while trail['timestamps'] and current_time - trail['timestamps'][0] > self.fade_duration:
                trail['points'].popleft()
                trail['timestamps'].popleft()

    def process_hand_landmarks(self, landmarks, frame_shape):
        """
        Extract index finger tip position from hand landmarks.

        Args:
            landmarks: Hand landmarks from mediapipe tasks
            frame_shape: Shape of the frame (height, width, channels)

        Returns:
            Tuple of (x, y) coordinates
        """
        h, w, _ = frame_shape
        # Index finger tip is landmark 8
        index_tip = landmarks[8]
        x = int(index_tip.x * w)
        y = int(index_tip.y * h)
        return x, y

    def create_hand_mask(self, hand_landmarks_list, frame_shape):
        """
        Create a binary mask showing only the hand regions.

        Args:
            hand_landmarks_list: List of hand landmarks from MediaPipe tasks
            frame_shape: Shape of the frame (height, width, channels)

        Returns:
            Binary mask with hand regions as white (255)
        """
        h, w, _ = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        for hand_landmarks in hand_landmarks_list:
            # Extract all landmark points
            points = []
            for landmark in hand_landmarks:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                points.append([x, y])

            # Create convex hull around hand landmarks
            points = np.array(points, dtype=np.int32)
            hull = cv2.convexHull(points)

            # Fill the convex hull on the mask
            cv2.fillConvexPoly(mask, hull, 255)

            # Dilate to make the mask slightly larger (smoother edges)
            kernel = np.ones((15, 15), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        return mask

    def draw_ui(self, frame):
        """Draw UI elements on the frame."""
        h, w = frame.shape[:2]

        # Status text
        status = "Drawing: ON" if self.drawing_enabled else "Drawing: OFF"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 0) if self.drawing_enabled else (0, 0, 255), 2)

        # Eraser status
        eraser_status = "Eraser: ON" if self.eraser_enabled else "Eraser: OFF"
        cv2.putText(frame, eraser_status, (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 0, 255) if self.eraser_enabled else (128, 128, 128), 2)

        # Hand isolation status
        isolation_status = "Hand Only: ON" if self.hand_isolation_enabled else "Hand Only: OFF"
        cv2.putText(frame, isolation_status, (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                   0.7, (0, 255, 255) if self.hand_isolation_enabled else (128, 128, 128), 2)

        # Instructions
        instructions = [
            "SPACE - Toggle drawing",
            "E - Toggle eraser",
            "H - Toggle hand isolation",
            "C - Clear canvas",
            "Q - Quit"
        ]

        y_offset = h - 115
        for instruction in instructions:
            cv2.putText(frame, instruction, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25

    def run(self):
        """Main application loop."""
        # Initialize video capture
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open webcam")
            return

        print("Hand Motion Drawing Started!")
        print("- Point with your index finger to draw")
        print("- SPACE: Toggle drawing on/off")
        print("- E: Toggle eraser mode")
        print("- H: Toggle hand isolation (show only hand, hide background)")
        print("- C: Clear canvas")
        print("- Q: Quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Initialize canvas if needed
            if self.canvas is None:
                self.canvas = np.zeros_like(frame)

            # Convert to RGB for MediaPipe (new tasks API needs Image)
            mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Process hands using new tasks API
            detection_result = self.hand_landmarker.detect(mi)

            # Extract hand landmarks from new format
            hand_landmarks_list = []
            if detection_result.hand_landmarks:
                for landmarks in detection_result.hand_landmarks:
                    hand_landmarks_list.append(landmarks)

            # Create isolated hand display if enabled
            if self.hand_isolation_enabled and hand_landmarks_list:
                # Create mask for hand regions
                mask = self.create_hand_mask(hand_landmarks_list, frame.shape)

                # Create black background
                isolated_frame = np.zeros_like(frame)

                # Apply mask to show only hand regions
                isolated_frame = cv2.bitwise_and(frame, frame, mask=mask)

                # Replace frame with isolated version
                frame = isolated_frame

            if hand_landmarks_list:
                for hand_idx, hand_landmarks in enumerate(hand_landmarks_list):
                    # Draw hand skeleton (manually since mp_draw isn't available in new API)
                    for connection in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                                      (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                                      (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17), (17, 5)]:
                        start, end = connection
                        if start < len(hand_landmarks) and end < len(hand_landmarks):
                            x0 = int(hand_landmarks[start].x * frame.shape[1])
                            y0 = int(hand_landmarks[start].y * frame.shape[0])
                            x1 = int(hand_landmarks[end].x * frame.shape[1])
                            y1 = int(hand_landmarks[end].y * frame.shape[0])
                            cv2.line(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)

                    # Get index finger tip position
                    x, y = self.process_hand_landmarks(hand_landmarks, frame.shape)

                    # Get additional finger positions for eraser multi-point support
                    # Landmarks: 8=index tip, 12=middle tip, 16=ring tip
                    finger_points = []
                    if self.eraser_enabled:
                        # Extract middle and ring finger tips
                        middle_landmark = hand_landmarks[12]
                        ring_landmark = hand_landmarks[16]

                        middle_x = int(middle_landmark.x * frame.shape[1])
                        middle_y = int(middle_landmark.y * frame.shape[0])

                        ring_x = int(ring_landmark.x * frame.shape[1])
                        ring_y = int(ring_landmark.y * frame.shape[0])

                        finger_points = [(middle_x, middle_y), (ring_x, ring_y)]
                    
                    # Draw a marker at finger tip
                    cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

                    # Add to trail if drawing is enabled
                    if self.drawing_enabled:
                        self.add_trail_point(x, y, hand_idx, finger_points if self.eraser_enabled else None)

            # Clean old points periodically
            self.clean_old_points()

            # Draw trails with fading effect
            self.draw_trails(frame)

            # Draw UI
            self.draw_ui(frame)

            # Display
            cv2.imshow('Hand Motion Drawing', frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord(' '):
                self.drawing_enabled = not self.drawing_enabled
                print(f"Drawing {'enabled' if self.drawing_enabled else 'disabled'}")
            elif key == ord('e') or key == ord('E'):
                self.eraser_enabled = not self.eraser_enabled
                print(f"Eraser {'enabled' if self.eraser_enabled else 'disabled'}")
            elif key == ord('h') or key == ord('H'):
                self.hand_isolation_enabled = not self.hand_isolation_enabled
                print(f"Hand isolation {'enabled' if self.hand_isolation_enabled else 'disabled'}")
            elif key == ord('c'):
                self.trails.clear()
                self.last_position.clear()
                self.canvas = np.zeros_like(self.canvas) if self.canvas is not None else None
                print("Canvas cleared")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()


def main():
    """Entry point for the application."""
    app = HandDrawing(fade_duration=3.0, trail_thickness=10)
    app.run()


if __name__ == "__main__":
    main()
