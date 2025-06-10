# Assistive Vision - User Manual

## Overview

Assistive Vision is an iOS application designed to help visually impaired individuals navigate their surroundings through real-time object detection and audio feedback. The app uses your iPhone's camera to identify objects and provides natural-sounding audio descriptions.

## Key Features

- **Real-time Object Detection**: Continuously scans your surroundings and identifies objects
- **Audio Feedback**: Provides clear, natural-sounding descriptions of detected objects
- **Accessibility Integration**: Works seamlessly with VoiceOver and Siri Shortcuts
- **Battery Optimization**: Includes a Battery Saver mode to extend device usage time
- **Privacy-Focused**: All processing happens on your device, ensuring your privacy

## Getting Started

### Installation
1. Run the installation through XCODE to install the app onto your iPhone
2. Open the app on your iPhone
3. Grant necessary permissions (camera, microphone, and notifications)
4. Configure Gemini API key in the Environment.swift file (obtain from https://ai.google.dev/gemini-api/docs/api-key)

### Initial Setup
1. Launch the app
2. Follow the on-screen instructions to complete the setup process
3. Configure your preferred voice settings and detection sensitivity
4. Enable VoiceOver integration if desired

## Using the App

### Basic Operation
1. Open the app
2. Point your device's camera at the scene you want to analyze
3. The app will automatically detect objects and provide audio feedback
4. Use the volume buttons to adjust the feedback volume

### Voice Commands
The app supports voice commands for hands-free operation. Here's how to use each type of command:

#### Basic Command Usage
1. Press and hold anywhere on the screen
2. While holding, speak your command clearly
3. Release your finger to send the command
4. Wait for the confirmation tone
5. The app will process your command and respond accordingly

#### Settings Commands

##### Changing Language
1. Hold the screen
2. Say either "Change language to english" or "Set language to english"
3. Release your finger
4. Wait for the confirmation tone
5. The app will switch to English and confirm the change

1. Hold the screen
2. Say either "Change language to spanish" or "Set language to spanish"
3. Release your finger
4. Wait for the confirmation tone
5. The app will switch to Spanish and confirm the change

##### Changing Detection Model
1. Hold the screen
2. Say one of:
   - "Set model to quick"
   - "Set model to optimal"
   - "Set model to intensive"
3. Release your finger
4. Wait for the confirmation tone
5. The app will switch to the selected model and confirm the change

##### Adjusting Haptic Feedback
1. Hold the screen
2. Say either:
   - "Enable haptics" or "Turn on haptics"
   - "Disable haptics" or "Turn off haptics"
3. Release your finger
4. Wait for the confirmation tone
5. The app will toggle haptic feedback and confirm the change

##### Adjusting Scanning Mode
1. Hold the screen
2. Say either:
   - "Enable continuous scanning" or "Turn on continuous scanning"
   - "Enable manual mode" or "Turn on manual mode"
3. Release your finger
4. Wait for the confirmation tone
5. The app will switch modes and confirm the change

##### Adjusting Detection Settings
1. Hold the screen
2. Say one of:
   - "Set items to [number]" (1-30)
   - "Set confidence to [number]" (1-10)
   - "Set IoU to [number]" (1-10)
3. Release your finger
4. Wait for the confirmation tone
5. The app will adjust the setting and confirm the change

#### Navigation Commands

##### Opening Menus
1. Hold the screen
2. Say one of:
   - "Open settings"
   - "Open help"
   - "Open FAQ"
3. Release your finger
4. Wait for the confirmation tone
5. The app will open the requested menu

##### Closing the App
1. Hold the screen
2. Say "Close the app"
3. Release your finger
4. Wait for the confirmation tone
5. The app will close

#### Detection Commands

##### Manual Scanning
1. Hold the screen
2. Say "Scan now"
3. Release your finger
4. Wait for the confirmation tone
5. The app will scan the environment and describe what it sees

##### Person Description
1. Hold the screen
2. Say "Describe this person"
3. Release your finger
4. Wait for the confirmation tone
5. The app will take a photo and provide a detailed description

##### Asking Questions (Gemini)
1. Hold the screen
2. Say "Ask" followed by your question
3. Release your finger
4. Wait for the confirmation tone
5. The app will process your question
6. Wait 2-3 seconds for the response
7. The response will be read out loud

#### Camera Commands

##### Switching Cameras
1. Hold the screen
2. Say "Switch camera"
3. Release your finger
4. Wait for the confirmation tone
5. The app will switch between front and back cameras and confirm the change

#### Double Tap Commands

##### Toggling Continuous Scanning
1. Double tap anywhere on the screen
2. The app will toggle continuous scanning mode
3. A confirmation message will be read out loud

#### Tips for Using Voice Commands
- Speak clearly and at a normal pace
- Ensure you're in a relatively quiet environment
- Hold your device at a comfortable distance from your mouth
- If a command isn't recognized, try rephrasing it
- Wait for the confirmation tone before speaking

### Advanced Features

#### Battery Saver Mode
1. Open the app settings
2. Toggle "Battery Saver Mode" on
3. The app will reduce frame capture rate to conserve battery

#### Siri Shortcuts
1. Open the Settings app
2. Navigate to Siri & Search
3. Add Assistive Vision shortcuts
4. Create custom voice commands for quick access

## Troubleshooting

### Common Issues

#### Camera Not Working
- Ensure camera permissions are granted
- Check for any physical obstructions
- Restart the app

#### No Audio Feedback
- Verify device volume is turned up
- Check if device is not in silent mode
- Ensure audio permissions are granted

#### Battery Drain
- Enable Battery Saver mode
- Reduce detection sensitivity
- Close other background apps

## Accessibility Features

### VoiceOver Integration
- The app is fully compatible with VoiceOver
- All buttons and controls are properly labeled
- Gesture controls are optimized for VoiceOver users

### Dynamic Type Support
- Text size adjusts according to system settings
- High contrast mode support
- Customizable interface colors

## Support

For additional help:
- Check out the HELP section for using the AI Assistant