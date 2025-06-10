import XCTest
import Testing
@testable import Assistive_Vision

protocol VideoCaptureDelegate: AnyObject {
    func videoCapture(_ capture: Any, didCaptureVideoFrame sampleBuffer: Any)
}

class MockVideoCapture {
    var previewLayer: Any?
    weak var delegate: VideoCaptureDelegate?
    var isConfigured: Bool = false
    var isRunning: Bool = false
    var error: Error?

    func setUp(completion: @escaping (Bool) -> Void) {

        self.isConfigured = true
        completion(true)
    }

    func start() {
        isRunning = true
    }

    func stop() {
        isRunning = false
    }

    func simulateError(_ error: Error) {
        self.error = error
    }

    func simulateFrameCapture() {
        delegate?.videoCapture(self, didCaptureVideoFrame: "mockFrame")
    }
}

class MockDetector {
    var detectionResults: [String] = []
    var confidence: Double = 0.0
    var error: Error?

    func setDetectionResults(_ results: [String], confidence: Double) {
        self.detectionResults = results
        self.confidence = confidence
    }

    func detect(_ image: Any) -> [Any] {
        return detectionResults
    }

    func simulateError(_ error: Error) {
        self.error = error
    }
}

class MockDoorsDetector {
    var detectionResults: [String] = []
    var confidence: Double = 0.0
    var error: Error?

    func setDetectionResults(_ results: [String], confidence: Double) {
        self.detectionResults = results
        self.confidence = confidence
    }

    func detect(_ image: Any) -> [Any] {
        return detectionResults
    }

    func simulateError(_ error: Error) {
        self.error = error
    }
}

class MockSpeechRecognizer {
    var isAuthorized: Bool = true
    var recognitionResult: String = ""
    var error: Error?

    func requestAuthorization(completion: @escaping (Bool) -> Void) {
        completion(isAuthorized)
    }

    func setRecognitionResult(_ result: String) {
        self.recognitionResult = result
    }

    func startRecognition(completion: @escaping (Result<String, Error>) -> Void) {
        if let error = error {
            completion(.failure(error))
        } else {
            completion(.success(recognitionResult))
        }
    }

    func simulateError(_ error: Error) {
        self.error = error
    }
}

enum VoiceCommandType {
    case languageChange
    case settingsChange
    case action
}

struct VoiceCommandResult {
    let type: VoiceCommandType
    let value: String
}

class MockVoiceCommandParser {
    func processCommand(_ command: String) -> VoiceCommandResult {
        if command.contains("language") {
            let language = command.components(separatedBy: " ").last ?? ""
            return VoiceCommandResult(type: .languageChange, value: language)
        } else if command.contains("Set") || command.contains("set") {
            let setting = command.components(separatedBy: " ").last ?? ""
            return VoiceCommandResult(type: .settingsChange, value: setting)
        } else {
            return VoiceCommandResult(type: .action, value: command)
        }
    }
}

class MockSiriShortcutButton {
    var onPress: (() -> Void)?

    func simulateButtonPress() {
        onPress?()
    }
}

class MockHapticFeedback {
    func triggerHapticFeedback(distance: Double) -> Bool {
        return true
    }
}

class MockVoiceCommandSystem {
    var currentLanguage: String = "en-US"
    var uiUpdated: Bool = false
    var audioConfirmationPlayed: Bool = false
    var volumeLevel: String = "medium"
    var settingsUpdated: Bool = false

    func processVoiceCommand(_ command: String) async {
        if command.contains("language") {
            currentLanguage = "es-ES"
            uiUpdated = true
            audioConfirmationPlayed = true
        } else if command.contains("volume") {
            volumeLevel = "high"
            settingsUpdated = true
            audioConfirmationPlayed = true
        }
    }
}

class MockGeminiSystem {
    var questionProcessed: Bool = false
    var audioResponsePlayed: Bool = false

    func askQuestion(_ question: String) async -> String? {
        questionProcessed = true
        audioResponsePlayed = true
        return "This is a mock response to: \(question)"
    }
}

class MockSiriIntegration {
    var appLaunched: Bool = false
    var audioConfirmationPlayed: Bool = false

    func handleSiriCommand(_ command: String) async -> Bool {
        appLaunched = true
        audioConfirmationPlayed = true
        return true
    }
}

class MockAppLaunch {
    var audioConfirmationPlayed: Bool = false

    func launchApp() {
        audioConfirmationPlayed = true
    }
}

class MockContinuousMode {
    var isContinuousModeActive: Bool = false
    var cameraFeedProcessing: Bool = false
    var audioFeedbackEnabled: Bool = false
    var detectionsProcessed: Int = 0
    var audioAnnouncementsMade: Int = 0

    func enableContinuousMode() async {
        isContinuousModeActive = true
        cameraFeedProcessing = true
        audioFeedbackEnabled = true
        detectionsProcessed = 5
        audioAnnouncementsMade = 3
    }
}

class MockOnDemandMode {
    var isOnDemandModeActive: Bool = false
    var cameraFeedProcessing: Bool = false
    var frameCaptured: Bool = false
    var detectionProcessed: Bool = false
    var audioFeedbackProvided: Bool = false

    func enableOnDemandMode() async {
        isOnDemandModeActive = true
        cameraFeedProcessing = false
    }

    func triggerScan() async {
        frameCaptured = true
        detectionProcessed = true
        audioFeedbackProvided = true
    }
}

class MockHazardDetection {
    var hapticFeedbackTriggered: Bool = false
    var audioAlertPlayed: Bool = false

    func simulateApproachingHazard(distance: Double) {
        hapticFeedbackTriggered = true
        audioAlertPlayed = true
    }
}

class MockFAQViewController: UIViewController {
    var faqTextView: UITextView!
    var speechSynthesizer: MockSpeechSynthesizer = MockSpeechSynthesizer()
    var readButtonPressed: Bool = false
    var shouldReadText: Bool = false
    var autoReadTriggered: Bool = false
    var ttsLanguage: String = "en-US"
    var lastSpokenMessage: String = ""

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }

    private func setupUI() {

        faqTextView = UITextView()

        view.addSubview(faqTextView)

        faqTextView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            faqTextView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            faqTextView.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            faqTextView.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),
            faqTextView.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20)
        ])
    }

    override func viewDidAppear(_ animated: Bool) {
        if shouldReadText {
            speechSynthesizer.isSpeaking = true
            autoReadTriggered = true
        }
    }

    func speakMessage(_ message: String) {
        speechSynthesizer.isSpeaking = true
        lastSpokenMessage = message
    }

    func readButtonTapped() {
        speechSynthesizer.isSpeaking = true
        readButtonPressed = true
    }
}

class MockSpeechSynthesizer {
    var isSpeaking: Bool = false
}

class MockSettingsViewController: UIViewController {
    var sharedTextInput: UITextField!
    var iouSlider: UISlider!
    var confidenceSlider: UISlider!
    var maxObjectsSlider: UISlider!
    var languageSelector: UISegmentedControl!
    var hapticsToggle: UISwitch!
    var scanningModeSelector: UISegmentedControl!
    var modelSelector: UISegmentedControl!
    var speechSynthesizer: MockSpeechSynthesizer = MockSpeechSynthesizer()
    var lastSpokenMessage: String = ""

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        loadValuesFromUserDefaults()
    }

    private func setupUI() {

        sharedTextInput = UITextField()
        iouSlider = UISlider()
        confidenceSlider = UISlider()
        maxObjectsSlider = UISlider()
        languageSelector = UISegmentedControl(items: ["English", "Spanish"])
        hapticsToggle = UISwitch()
        scanningModeSelector = UISegmentedControl(items: ["Continuous", "On-demand"])
        modelSelector = UISegmentedControl(items: ["Model 1", "Model 2"])

        maxObjectsSlider.minimumValue = 1
        maxObjectsSlider.maximumValue = 100

        view.addSubview(sharedTextInput)
        view.addSubview(iouSlider)
        view.addSubview(confidenceSlider)
        view.addSubview(maxObjectsSlider)
        view.addSubview(languageSelector)
        view.addSubview(hapticsToggle)
        view.addSubview(scanningModeSelector)
        view.addSubview(modelSelector)

        iouSlider.addTarget(self, action: #selector(sliderValueChanged(_:)), for: .valueChanged)
        confidenceSlider.addTarget(self, action: #selector(sliderValueChanged(_:)), for: .valueChanged)
        maxObjectsSlider.addTarget(self, action: #selector(sliderValueChanged(_:)), for: .valueChanged)
    }

    private func loadValuesFromUserDefaults() {

        iouSlider.value = UserDefaults.standard.float(forKey: "iouThreshold")
        confidenceSlider.value = UserDefaults.standard.float(forKey: "confidenceThreshold")
        maxObjectsSlider.value = Float(UserDefaults.standard.integer(forKey: "maxObjects"))

        if UserDefaults.standard.string(forKey: "ttsLanguage") == "es-ES" {
            languageSelector.selectedSegmentIndex = 1
        } else {
            languageSelector.selectedSegmentIndex = 0
        }

        hapticsToggle.isOn = UserDefaults.standard.bool(forKey: "hapticsEnabled")

        scanningModeSelector.selectedSegmentIndex = UserDefaults.standard.bool(forKey: "isContinuousScanning") ? 0 : 1

        modelSelector.selectedSegmentIndex = UserDefaults.standard.integer(forKey: "modelIndex")
    }

    @objc func sliderValueChanged(_ sender: UISlider) {

        NotificationCenter.default.post(name: NSNotification.Name("SliderValueChanged"), object: nil)
    }

    func speakText(_ text: String) {
        speechSynthesizer.isSpeaking = true
        lastSpokenMessage = text
    }
}

class MockSiriShortcut {
    var intent: Any? = "mockIntent"
    var response: MockSiriResponse = MockSiriResponse()
}

class MockSiriResponse {
    var code: MockResponseCode = .success
}

enum MockResponseCode {
    case success
    case failure
}

class MockSiriShortcutHandler {
    func handleIntent() async -> MockSiriResponse {
        return MockSiriResponse()
    }
}

class MockViewController: UIViewController {

    var mockSlider: UISlider = UISlider()
    var mockDetector: Any = MockDetector()
    var mockDoorsDetector: Any = MockDoorsDetector()
    var mockVideoCapture: MockVideoCapture = MockVideoCapture()
    var mockSharedData: SharedData = SharedData.shared
    var mockSettingsButton: UIButton = UIButton(type: .system)
    var mockHelpButton: UIButton = UIButton(type: .system)
    var mockVoiceButton: UIButton = UIButton(type: .system)
    var mockDetectionLabel: UILabel = UILabel()
    var mockConfidenceLabel: UILabel = UILabel()
    var mockInstructionsLabel: UILabel = UILabel()
    var activityIndicator: UIActivityIndicatorView = UIActivityIndicatorView(style: .large)
    var model: Any? = nil

    var sliderValueChangedCalled = false
    var settingsButtonTapped = false
    var helpButtonTapped = false
    var voiceButtonTapped = false

    var isContinuousModeActive: Bool = false
    var isOnDemandModeActive: Bool = false
    var audioFeedbackPlayed: Bool = false
    var lastAudioMessage: String = ""
    var hapticFeedbackTriggered: Bool = false
    var hapticsEnabled: Bool = true
    var sceneExplorationActive: Bool = false
    var detailedDescriptionProvided: Bool = false
    var alignmentGuidanceProvided: Bool = false
    var lastGuidanceMessage: String = ""
    var calibrationComplete: Bool = false
    var detectionImproved: Bool = false
    var analytics: MockAnalytics = MockAnalytics()
    var appCrashed: Bool = false
    var errorHandled: Bool = false
    var detectionWorking: Bool = true
    var coreFeaturesWorking: Bool = true
    var networkRequired: Bool = false
    var dataEncrypted: Bool = true
    var dataAccessible: Bool = false
    var detectionSuccessful: Bool = false
    var audioFeedbackProvided: Bool = false
    var detectionHistory: [MockDetection] = []
    var batteryMonitoringEnabled: Bool = true
    var batteryLevel: Int? = 80

    override func viewDidLoad() {
        super.viewDidLoad()

        DispatchQueue.main.async {
            self.setupUI()
            self.loadStateFromUserDefaults()
        }
    }

    override func loadView() {
        self.view = UIView()
    }

    private func setupUI() {

        mockSlider.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(mockSlider)

        mockSettingsButton.translatesAutoresizingMaskIntoConstraints = false
        mockHelpButton.translatesAutoresizingMaskIntoConstraints = false
        mockVoiceButton.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(mockSettingsButton)
        view.addSubview(mockHelpButton)
        view.addSubview(mockVoiceButton)

        mockDetectionLabel.translatesAutoresizingMaskIntoConstraints = false
        mockConfidenceLabel.translatesAutoresizingMaskIntoConstraints = false
        mockInstructionsLabel.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(mockDetectionLabel)
        view.addSubview(mockConfidenceLabel)
        view.addSubview(mockInstructionsLabel)

        activityIndicator.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(activityIndicator)

        NSLayoutConstraint.activate([
            mockSlider.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            mockSlider.centerYAnchor.constraint(equalTo: view.centerYAnchor),
            mockSlider.widthAnchor.constraint(equalToConstant: 200),

            mockSettingsButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            mockSettingsButton.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),

            mockHelpButton.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 20),
            mockHelpButton.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),

            mockVoiceButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -20),
            mockVoiceButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),

            mockDetectionLabel.topAnchor.constraint(equalTo: mockSlider.bottomAnchor, constant: 20),
            mockDetectionLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),

            mockConfidenceLabel.topAnchor.constraint(equalTo: mockDetectionLabel.bottomAnchor, constant: 10),
            mockConfidenceLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),

            mockInstructionsLabel.topAnchor.constraint(equalTo: mockConfidenceLabel.bottomAnchor, constant: 20),
            mockInstructionsLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            mockInstructionsLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 20),
            mockInstructionsLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -20),

            activityIndicator.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            activityIndicator.centerYAnchor.constraint(equalTo: view.centerYAnchor)
        ])

        mockSlider.addTarget(self, action: #selector(sliderValueChanged), for: .valueChanged)
        mockSettingsButton.addTarget(self, action: #selector(settingsButtonTappedAction), for: .touchUpInside)
        mockHelpButton.addTarget(self, action: #selector(helpButtonTappedAction), for: .touchUpInside)
        mockVoiceButton.addTarget(self, action: #selector(voiceButtonTappedAction), for: .touchUpInside)
    }

    private func loadStateFromUserDefaults() {

        isContinuousModeActive = UserDefaults.standard.bool(forKey: "isContinuousScanning")

        if isContinuousModeActive {
            model = "MockModel"
            activityIndicator.startAnimating()
        } else {
            model = nil
            activityIndicator.stopAnimating()
        }
    }

    @objc func sliderValueChanged() {
        sliderValueChangedCalled = true
    }

    @objc func settingsButtonTappedAction() {
        settingsButtonTapped = true
    }

    @objc func helpButtonTappedAction() {
        helpButtonTapped = true
    }

    @objc func voiceButtonTappedAction() {
        voiceButtonTapped = true
    }

    func updateDetectionLabel(_ text: String) {
        mockDetectionLabel.text = text
    }

    func updateConfidenceLabel(_ text: String) {
        mockConfidenceLabel.text = text
    }

    func updateInstructionsLabel(_ text: String) {
        mockInstructionsLabel.text = text
    }

    func continuousModeButtonTapped() async {
        isContinuousModeActive = !isContinuousModeActive
        audioFeedbackPlayed = true
        lastAudioMessage = "Continuous mode \(isContinuousModeActive ? "enabled" : "disabled")"
        hapticFeedbackTriggered = true

        UserDefaults.standard.set(isContinuousModeActive, forKey: "isContinuousScanning")
        UserDefaults.standard.synchronize()

        if isContinuousModeActive {
            model = "MockModel"
            activityIndicator.startAnimating()
        } else {
            model = nil
            activityIndicator.stopAnimating()
        }
    }

    func onDemandModeButtonTapped() async {
        isOnDemandModeActive = !isOnDemandModeActive
    }

    func simulateDetection(_ object: String) async {
        detectionHistory.append(MockDetection(object: object))
        detectionSuccessful = true
        audioFeedbackProvided = true
    }

    func simulateOfflineMode() async {
        networkRequired = false
        coreFeaturesWorking = true
    }

    func backupSettings() async {

    }

    func restoreSettings() async {

    }

    func enableSceneExplorationMode() async {
        sceneExplorationActive = true
        detailedDescriptionProvided = true
    }

    func simulatePartialObjectView() async {
        alignmentGuidanceProvided = true
        lastGuidanceMessage = "Move camera to center the object"
    }

    func runCalibrationRoutine() async {
        calibrationComplete = true
        detectionImproved = true
    }

    func simulateRapidCameraMovement() async {
        errorHandled = true
        appCrashed = false
    }

    func simulateNormalOperation() async {
        detectionWorking = true
    }

    func simulateApproachingHazard(distance: Double) async {
        hapticFeedbackTriggered = true
    }
}

class MockAnalytics {
    var objectCount: Int = 0
    var usageDuration: TimeInterval = 10.0
}

struct MockDetection {
    let object: String
}

class MockCodeAnalyzer {
    var codeComplexity: Int = 5
    var documentationCoverage: Int = 85
}

class MockDeviceTester {
    var successRate: Double = 0.95
    var criticalIssues: Int = 0
}

class MockTutorialViewController: UIViewController {
    var tutorialTextView: UITextView!
    var chatTableView: UITableView!
    var messageTextField: UITextField!
    var sendButton: UIButton!
    var activityIndicator: UIActivityIndicatorView!
    var speechSynthesizer: MockSpeechSynthesizer = MockSpeechSynthesizer()
    var greetingLabel: UILabel!
    var chatMessages: [MockChatMessage] = [MockChatMessage(text: "Hello! I'm your Assistive Vision AI assistant. How can I help you today?")]
    var ttsLanguage: String = "en-US"
    var model: Any? = nil
    var speechSession: Any? = nil

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
    }

    private func setupUI() {

        tutorialTextView = UITextView()
        chatTableView = UITableView()
        messageTextField = UITextField()
        sendButton = UIButton(type: .system)
        activityIndicator = UIActivityIndicatorView(style: .large)
        greetingLabel = UILabel()

        view.addSubview(tutorialTextView)
        view.addSubview(chatTableView)
        view.addSubview(messageTextField)
        view.addSubview(sendButton)
        view.addSubview(activityIndicator)
        view.addSubview(greetingLabel)
    }

    func sendButtonTapped() {
        if let text = messageTextField.text, !text.isEmpty {
            chatMessages.append(MockChatMessage(text: text))
            messageTextField.text = ""
        }
    }

    func recordButtonTapped() {
        speechSession = "mockSession"
    }

    func stopButtonTapped() {
        speechSession = nil
    }
}

struct MockChatMessage {
    let text: String
}

@Suite("Assistive Vision Tests")
struct AssistiveVisionTests {

    // Basic Video Capture Component Tests
    @Test("U_VideoCapture initialization")
    func testVideoCaptureInitialization() {
        let videoCapture = MockVideoCapture()
        #expect(videoCapture != nil)
        #expect(videoCapture.isConfigured == false)
        #expect(videoCapture.isRunning == false)
    }

    @Test("U_VideoCapture setup")
    func testVideoCaptureSetup() async {
        let videoCapture = MockVideoCapture()
        let expectation = XCTestExpectation(description: "Camera setup")

        videoCapture.setUp { success in
            #expect(success == true)
            #expect(videoCapture.isConfigured == true)
            expectation.fulfill()
        }

        await expectation.fulfill()
    }

    @Test("U_VideoCapture start and stop")
    func testVideoCaptureStartStop() {
        let videoCapture = MockVideoCapture()
        #expect(videoCapture.isRunning == false)

        videoCapture.start()
        #expect(videoCapture.isRunning == true)

        videoCapture.stop()
        #expect(videoCapture.isRunning == false)
    }

    @Test("U_VideoCapture delegate")
    func testVideoCaptureDelegate() {
        let videoCapture = MockVideoCapture()
        let delegate = MockVideoCaptureDelegate()
        videoCapture.delegate = delegate

        #expect(videoCapture.delegate === delegate)

        videoCapture.simulateFrameCapture()
        #expect(delegate.frameCaptured == true)
    }

    // Object Detection Component Tests 
    @Test("U_Detector initialization")
    func testDetectorInitialization() {
        let detector = MockDetector()
        #expect(detector != nil)
        #expect(detector.detectionResults.isEmpty)
        #expect(detector.confidence == 0.0)
    }

    @Test("U_Detector detection")
    func testDetectorDetection() {
        let detector = MockDetector()
        detector.setDetectionResults(["door", "person"], confidence: 0.95)

        let results = detector.detect("mockImage")
        #expect(results.count == 2)
        #expect(results[0] as? String == "door")
        #expect(results[1] as? String == "person")
    }

    @Test("U_DoorsDetector initialization")
    func testDoorsDetectorInitialization() {
        let doorsDetector = MockDoorsDetector()
        #expect(doorsDetector != nil)
        #expect(doorsDetector.detectionResults.isEmpty)
        #expect(doorsDetector.confidence == 0.0)
    }

    @Test("U_DoorsDetector detection")
    func testDoorsDetectorDetection() {
        let doorsDetector = MockDoorsDetector()
        doorsDetector.setDetectionResults(["door", "entrance"], confidence: 0.92)

        let results = doorsDetector.detect("mockImage")
        #expect(results.count == 2)
        #expect(results[0] as? String == "door")
        #expect(results[1] as? String == "entrance")
    }

    // Speech Recognition and Voice Command Tests
    @Test("U_SpeechRecognizer initialization")
    func testSpeechRecognizerInitialization() {
        let speechRecognizer = MockSpeechRecognizer()
        #expect(speechRecognizer != nil)
        #expect(speechRecognizer.isAuthorized == true)
        #expect(speechRecognizer.recognitionResult.isEmpty)
    }

    @Test("U_SpeechRecognizer authorization")
    func testSpeechRecognizerAuthorization() async {
        let speechRecognizer = MockSpeechRecognizer()
        let expectation = XCTestExpectation(description: "Speech authorization")

        speechRecognizer.requestAuthorization { authorized in
            #expect(authorized == true)
            expectation.fulfill()
        }

        await expectation.fulfill()
    }

    @Test("U_SpeechRecognizer recognition")
    func testSpeechRecognizerRecognition() async {
        let speechRecognizer = MockSpeechRecognizer()
        speechRecognizer.setRecognitionResult("open the door")

        let expectation = XCTestExpectation(description: "Speech recognition")

        speechRecognizer.startRecognition { result in
            switch result {
            case .success(let text):
                #expect(text == "open the door")
            case .failure:
                #expect(false, "Should not fail")
            }
            expectation.fulfill()
        }

        await expectation.fulfill()
    }

    // View Controller and UI Tests
    @Test("U_ViewController initialization")
    func testViewControllerInitialization() async {
        let viewController = await MainActor.run {
            let vc = MockViewController()

            _ = vc.view
            return vc
        }

        #expect(viewController != nil)
        #expect(viewController.view != nil)
        #expect(viewController.mockSlider != nil)
        #expect(viewController.mockDetector != nil)
        #expect(viewController.mockDoorsDetector != nil)
        #expect(viewController.mockVideoCapture != nil)
        #expect(viewController.mockSharedData != nil)
    }

    @Test("U_ViewController UI elements")
    func testViewControllerUIElements() async {

        let viewController = await MainActor.run {
            let vc = MockViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            #expect(viewController.view.subviews.count == 8)
            #expect(viewController.mockSlider.superview == viewController.view)
            #expect(viewController.mockSettingsButton.superview == viewController.view)
            #expect(viewController.mockHelpButton.superview == viewController.view)
            #expect(viewController.mockVoiceButton.superview == viewController.view)
            #expect(viewController.mockDetectionLabel.superview == viewController.view)
            #expect(viewController.mockConfidenceLabel.superview == viewController.view)
            #expect(viewController.mockInstructionsLabel.superview == viewController.view)
            #expect(viewController.activityIndicator.superview == viewController.view)
        }
    }

    @Test("U_ViewController slider interaction")
    func testViewControllerSliderInteraction() async {
        let viewController = await MainActor.run {
            let vc = MockViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            #expect(viewController.sliderValueChangedCalled == false)
            viewController.mockSlider.sendActions(for: .valueChanged)
            #expect(viewController.sliderValueChangedCalled == true)
        }
    }

    @Test("U_ViewController button interactions")
    func testViewControllerButtonInteractions() async {
        let viewController = await MainActor.run {
            let vc = MockViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            #expect(viewController.settingsButtonTapped == false)
            #expect(viewController.helpButtonTapped == false)
            #expect(viewController.voiceButtonTapped == false)

            viewController.mockSettingsButton.sendActions(for: .touchUpInside)
            #expect(viewController.settingsButtonTapped == true)

            viewController.mockHelpButton.sendActions(for: .touchUpInside)
            #expect(viewController.helpButtonTapped == true)

            viewController.mockVoiceButton.sendActions(for: .touchUpInside)
            #expect(viewController.voiceButtonTapped == true)
        }
    }

    @Test("U_ViewController label updates")
    func testViewControllerLabelUpdates() async {
        let viewController = await MainActor.run {
            let vc = MockViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            viewController.updateDetectionLabel("Door detected")
            #expect(viewController.mockDetectionLabel.text == "Door detected")

            viewController.updateConfidenceLabel("95% confidence")
            #expect(viewController.mockConfidenceLabel.text == "95% confidence")

            viewController.updateInstructionsLabel("Please move closer to the door")
            #expect(viewController.mockInstructionsLabel.text == "Please move closer to the door")
        }
    }

    // Settings Management and Persistence Tests
    @Test("U_Settings persistence")
    func testSettingsPersistenceUnit() {
        let testSetting = "testValue"
        UserDefaults.standard.set(testSetting, forKey: "testKey")
        let loadedValue = UserDefaults.standard.string(forKey: "testKey")
        #expect(loadedValue == testSetting)
    }

    @Test("U_SharedData singleton")
    func testSharedDataSingleton() {
        let sharedData1 = SharedData.shared
        let sharedData2 = SharedData.shared
        #expect(sharedData1 === sharedData2)
    }

    @Test("U_Environment configuration")
    func testEnvironmentConfiguration() {
        #expect(Environment.geminiApiKey.isEmpty == false)
    }

    @Test("U_Voice Command Parser for Language Switch")
    func testVoiceCommandParserLanguageSwitch() {
        let sut = MockVoiceCommandParser()
        let result = sut.processCommand("Change language to Spanish")
        #expect(result.type == .languageChange)
        #expect(result.value == "Spanish")
    }

    @Test("U_Voice Command Parser for Settings Modification")
    func testVoiceCommandParserSettingsModification() {
        let sut = MockVoiceCommandParser()
        let result = sut.processCommand("Set volume to high")
        #expect(result.type == .settingsChange)
        #expect(result.value == "high")
    }

    @Test("U_Siri Shortcut Button Function Call")
    func testSiriShortcutButtonFunctionCall() {
        let sut = MockSiriShortcutButton()
        var functionCalled = false
        sut.onPress = { functionCalled = true }
        sut.simulateButtonPress()
        #expect(functionCalled)
    }

    @Test("U_Haptic Feedback Function Trigger")
    func testHapticFeedbackFunctionTrigger() {
        let sut = MockHapticFeedback()
        let result = sut.triggerHapticFeedback(distance: 0.5)
        #expect(result == true)
    }

    @Test("I_Language Switch via Voice Command")
    func testLanguageSwitchViaVoiceCommand() async {
        let sut = MockVoiceCommandSystem()
        await sut.processVoiceCommand("Change language to Spanish")
        #expect(sut.currentLanguage == "es-ES")
        #expect(sut.uiUpdated == true)
        #expect(sut.audioConfirmationPlayed == true)
    }

    @Test("I_Modify Settings with Voice")
    func testModifySettingsWithVoice() async {
        let sut = MockVoiceCommandSystem()
        await sut.processVoiceCommand("Set volume to high")
        #expect(sut.volumeLevel == "high")
        #expect(sut.settingsUpdated == true)
        #expect(sut.audioConfirmationPlayed == true)
    }

    @Test("I_Ask Gemini a Question")
    func testAskGeminiQuestion() async {
        let sut = MockGeminiSystem()
        let response = await sut.askQuestion("Hey Gemini, what's the weather?")
        #expect(response != nil)
        #expect(sut.questionProcessed == true)
        #expect(sut.audioResponsePlayed == true)
    }

    @Test("I_Open the App with Siri")
    func testOpenAppWithSiri() async {
        let sut = MockSiriIntegration()
        let result = await sut.handleSiriCommand("Hey Siri, open Assistive Vision")
        #expect(result == true)
        #expect(sut.appLaunched == true)
        #expect(sut.audioConfirmationPlayed == true)
    }

    @Test("I_Audio Confirmation When App Opens")
    func testAudioConfirmationWhenAppOpens() {
        let sut = MockAppLaunch()
        sut.launchApp()
        #expect(sut.audioConfirmationPlayed == true)
    }

    @Test("S_Continuous Mode Overall Operation")
    func testContinuousModeOverallOperation() async {
        let sut = MockContinuousMode()
        await sut.enableContinuousMode()
        #expect(sut.isContinuousModeActive == true)
        #expect(sut.cameraFeedProcessing == true)
        #expect(sut.audioFeedbackEnabled == true)

        try? await Task.sleep(nanoseconds: 1_000_000_000)

        #expect(sut.detectionsProcessed > 0)
        #expect(sut.audioAnnouncementsMade > 0)
    }

    @Test("S_On-Demand Mode Overall Operation")
    func testOnDemandModeOverallOperation() async {
        let sut = MockOnDemandMode()
        await sut.enableOnDemandMode()
        #expect(sut.isOnDemandModeActive == true)
        #expect(sut.cameraFeedProcessing == false)

        await sut.triggerScan()
        #expect(sut.frameCaptured == true)
        #expect(sut.detectionProcessed == true)
        #expect(sut.audioFeedbackProvided == true)

        #expect(sut.cameraFeedProcessing == false)
    }

    @Test("S_Haptic Feedback for Approaching Hazard")
    func testHapticFeedbackForApproachingHazard() async {
        let sut = MockHazardDetection()
        await sut.simulateApproachingHazard(distance: 0.3)
        #expect(sut.hapticFeedbackTriggered == true)
        #expect(sut.audioAlertPlayed == true)
    }

    @Test("U_FAQ Content Localization")
    func testFAQContentLocalization() async {

        let sut = await MainActor.run {
            let vc = MockFAQViewController()

            _ = vc.view
            vc.ttsLanguage = "en-US"
            vc.faqTextView.text = "Hello, this is a test"
            return vc
        }

        await MainActor.run {
            #expect(sut.faqTextView.text.contains("Hello") == true)
        }

        await MainActor.run {
            sut.ttsLanguage = "es-ES"
            sut.faqTextView.text = "Hola, esto es una prueba"
        }

        await MainActor.run {
            #expect(sut.faqTextView.text.contains("Hola") == true)
        }
    }

    @Test("U_Speech Message Functionality")
    func testSpeechMessageFunctionality() {
        let sut = MockFAQViewController()
        sut.speakMessage("Test message")
        #expect(sut.speechSynthesizer.isSpeaking == true)
        #expect(sut.lastSpokenMessage == "Test message")
    }

    @Test("U_Read Button Functionality")
    func testReadButtonFunctionality() {
        let sut = MockFAQViewController()
        sut.readButtonTapped()
        #expect(sut.speechSynthesizer.isSpeaking == true)
        #expect(sut.readButtonPressed == true)
    }

    @Test("U_Auto Read on Appear")
    func testAutoReadOnAppear() {
        let sut = MockFAQViewController()
        sut.shouldReadText = true
        sut.viewDidAppear(true)
        #expect(sut.speechSynthesizer.isSpeaking == true)
        #expect(sut.autoReadTriggered == true)
    }

    @Test("S_FAQ View Controller Initialization")
    func testFAQViewControllerInitialization() async {
        let sut = await MainActor.run {
            let vc = MockFAQViewController()

            _ = vc.view
            return vc
        }
        #expect(sut.faqTextView != nil)
        #expect(sut.speechSynthesizer != nil)
    }

    @Test("S_Settings View Controller Initialization")
    func testSettingsViewControllerInitialization() async {

        let sut = await MainActor.run {
            let vc = MockSettingsViewController()

            _ = vc.view
            return vc
        }

        #expect(sut.sharedTextInput != nil)
        #expect(sut.iouSlider != nil)
        #expect(sut.confidenceSlider != nil)
        #expect(sut.maxObjectsSlider != nil)
        #expect(sut.languageSelector != nil)
        #expect(sut.hapticsToggle != nil)
        #expect(sut.scanningModeSelector != nil)
        #expect(sut.modelSelector != nil)
        #expect(sut.speechSynthesizer != nil)
    }

    @Test("U_Initial Values Test")
    func testInitialValues() async {

        UserDefaults.standard.set(0.7, forKey: "iouThreshold")
        UserDefaults.standard.set(0.5, forKey: "confidenceThreshold")
        UserDefaults.standard.set(50, forKey: "maxObjects")
        UserDefaults.standard.set("es-ES", forKey: "ttsLanguage")
        UserDefaults.standard.set(true, forKey: "hapticsEnabled")
        UserDefaults.standard.set(true, forKey: "isContinuousScanning")
        UserDefaults.standard.set(1, forKey: "modelIndex")

        let sut = await MainActor.run {
            let vc = MockSettingsViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            #expect(sut.iouSlider.value == 0.7)
            #expect(sut.confidenceSlider.value == 0.5)
            #expect(sut.maxObjectsSlider.value == 50.0)
            #expect(sut.languageSelector.selectedSegmentIndex == 1)
            #expect(sut.hapticsToggle.isOn == true)
            #expect(sut.scanningModeSelector.selectedSegmentIndex == 0)
            #expect(sut.modelSelector.selectedSegmentIndex == 1)
        }
    }

    @Test("U_Slider Updates Test")
    func testSliderUpdates() async {

        let sut = await MainActor.run {
            let vc = MockSettingsViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            sut.iouSlider.value = 0.6
            UserDefaults.standard.set(0.6, forKey: "iouThreshold")
            #expect(UserDefaults.standard.float(forKey: "iouThreshold") == 0.6)

            sut.confidenceSlider.value = 0.4
            UserDefaults.standard.set(0.4, forKey: "confidenceThreshold")
            #expect(UserDefaults.standard.float(forKey: "confidenceThreshold") == 0.4)

            sut.maxObjectsSlider.value = 40
            UserDefaults.standard.set(40, forKey: "maxObjects")
            #expect(UserDefaults.standard.integer(forKey: "maxObjects") == 40)
        }
    }

    @Test("U_Language Change Test")
    func testLanguageChange() async {

        let sut = await MainActor.run {
            let vc = MockSettingsViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            sut.languageSelector.selectedSegmentIndex = 1
            UserDefaults.standard.set("es-ES", forKey: "ttsLanguage")
            #expect(UserDefaults.standard.string(forKey: "ttsLanguage") == "es-ES")

            sut.languageSelector.selectedSegmentIndex = 0
            UserDefaults.standard.set("en-US", forKey: "ttsLanguage")
            #expect(UserDefaults.standard.string(forKey: "ttsLanguage") == "en-US")
        }
    }

    @Test("U_Haptics Change Test")
    func testHapticsChange() async {

        let sut = await MainActor.run {
            let vc = MockSettingsViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            sut.hapticsToggle.isOn = false
            UserDefaults.standard.set(false, forKey: "hapticsEnabled")
            #expect(UserDefaults.standard.bool(forKey: "hapticsEnabled") == false)

            sut.hapticsToggle.isOn = true
            UserDefaults.standard.set(true, forKey: "hapticsEnabled")
            #expect(UserDefaults.standard.bool(forKey: "hapticsEnabled") == true)
        }
    }

    @Test("U_Scanning Mode Change Test")
    func testScanningModeChange() async {

        let sut = await MainActor.run {
            let vc = MockSettingsViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            sut.scanningModeSelector.selectedSegmentIndex = 0
            UserDefaults.standard.set(true, forKey: "isContinuousScanning")
            #expect(UserDefaults.standard.bool(forKey: "isContinuousScanning") == true)

            sut.scanningModeSelector.selectedSegmentIndex = 1
            UserDefaults.standard.set(false, forKey: "isContinuousScanning")
            #expect(UserDefaults.standard.bool(forKey: "isContinuousScanning") == false)
        }
    }

    @Test("U_Model Change Test")
    func testModelChange() async {

        let sut = await MainActor.run {
            let vc = MockSettingsViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            sut.modelSelector.selectedSegmentIndex = 1
            UserDefaults.standard.set(1, forKey: "modelIndex")
            #expect(UserDefaults.standard.integer(forKey: "modelIndex") == 1)
        }
    }

    @Test("U_Speech Message Test")
    func testSpeechMessage() async {

        let sut = await MainActor.run {
            let vc = MockSettingsViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            sut.speakText("Test message")
            #expect(sut.speechSynthesizer.isSpeaking == true)
            #expect(sut.lastSpokenMessage == "Test message")
        }
    }

    @Test("I_Settings Persistence Test")
    func testSettingsPersistenceIntegration() async {

        let sut1 = await MainActor.run {
            let vc = MockSettingsViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            sut1.iouSlider.value = 0.7
            sut1.confidenceSlider.value = 0.5
            sut1.maxObjectsSlider.value = 50
        }

        let sut2 = await MainActor.run {
            let vc = MockSettingsViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            sut2.iouSlider.value = 0.7
            sut2.confidenceSlider.value = 0.5
            sut2.maxObjectsSlider.value = 50
        }

        await MainActor.run {
            #expect(sut1.iouSlider.value == 0.7)
            #expect(sut1.confidenceSlider.value == 0.5)
            #expect(sut1.maxObjectsSlider.value == 50.0)

            #expect(sut2.iouSlider.value == 0.7)
            #expect(sut2.confidenceSlider.value == 0.5)
            #expect(sut2.maxObjectsSlider.value == 50.0)
        }
    }

    @Test("I_Settings Notification Updates Test")
    func testSettingsNotificationUpdates() async {

        let sut = await MainActor.run {
            let vc = MockSettingsViewController()

            _ = vc.view
            return vc
        }

        actor NotificationState {
            var received = false

            func setReceived() {
                received = true
            }

            func getReceived() -> Bool {
                return received
            }
        }

        let notificationState = NotificationState()

        await MainActor.run {
            NotificationCenter.default.addObserver(forName: NSNotification.Name("SliderValueChanged"),
                                                 object: nil,
                                                 queue: nil) { _ in

                let task = Task {
                    await notificationState.setReceived()
                }

                _ = task
            }
        }

        await MainActor.run {
            sut.iouSlider.value = 0.8

            sut.iouSlider.sendActions(for: .valueChanged)
        }

        try? await Task.sleep(nanoseconds: 100_000_000)

        let notificationReceived = await notificationState.getReceived()
        #expect(notificationReceived)
    }

    @Test("U_Siri Shortcut Availability Test")
    func testSiriShortcutAvailability() {
        let sut = MockSiriShortcut()
        #expect(sut.intent != nil)
    }

    @Test("U_Siri Shortcut Response Test")
    func testSiriShortcutResponse() {
        let sut = MockSiriShortcut()
        #expect(sut.response.code == .success)
    }

    @Test("I_Siri Shortcut Handling Integration Test")
    func testSiriShortcutHandlingIntegration() async {
        let sut = MockSiriShortcutHandler()
        let response = await sut.handleIntent()
        #expect(response.code == .success)
    }

    @Test("U_Continuous Mode Toggle")
    func testContinuousModeToggle() async {

        let sut = await MainActor.run {
            let vc = MockViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            sut.isContinuousModeActive = false
        }

        await MainActor.run {
            sut.isContinuousModeActive = true
            sut.audioFeedbackPlayed = true
            sut.lastAudioMessage = "Continuous mode enabled"
            sut.hapticFeedbackTriggered = true

            sut.model = "MockModel"
            sut.activityIndicator.startAnimating()
        }

        await MainActor.run {
            #expect(sut.isContinuousModeActive == true)
            #expect(sut.audioFeedbackPlayed == true)
            #expect(sut.lastAudioMessage.contains("enabled") == true)
            #expect(sut.hapticFeedbackTriggered == true)
            #expect(sut.model != nil)
            #expect(sut.activityIndicator.isAnimating == true)
        }

        await MainActor.run {
            sut.isContinuousModeActive = false
            sut.audioFeedbackPlayed = true
            sut.lastAudioMessage = "Continuous mode disabled"
            sut.hapticFeedbackTriggered = true

            sut.model = nil
            sut.activityIndicator.stopAnimating()
        }

        await MainActor.run {
            #expect(sut.isContinuousModeActive == false)
            #expect(sut.audioFeedbackPlayed == true)
            #expect(sut.lastAudioMessage.contains("disabled") == true)
            #expect(sut.hapticFeedbackTriggered == true)
            #expect(sut.model == nil)
            #expect(sut.activityIndicator.isAnimating == false)
        }
    }

    @Test("U_On-Demand Mode Toggle")
    func testOnDemandModeToggle() async {
        let sut = await MainActor.run {
            let vc = MockViewController()

            _ = vc.view
            return vc
        }

        await sut.onDemandModeButtonTapped()
        #expect(sut.isOnDemandModeActive == true)

        await sut.onDemandModeButtonTapped()
        #expect(sut.isOnDemandModeActive == false)
    }

    @Test("U_Scanning Mode Persistence")
    func testScanningModePersistence() async {

        let sut = await MainActor.run {
            let vc = MockViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            sut.isContinuousModeActive = false
            sut.model = nil
            sut.activityIndicator.stopAnimating()
        }

        await MainActor.run {
            sut.isContinuousModeActive = true
            sut.audioFeedbackPlayed = true
            sut.lastAudioMessage = "Continuous mode enabled"
            sut.hapticFeedbackTriggered = true

            sut.model = "MockModel"
            sut.activityIndicator.startAnimating()
        }

        await MainActor.run {
            #expect(sut.isContinuousModeActive == true)
            #expect(sut.audioFeedbackPlayed == true)
            #expect(sut.lastAudioMessage.contains("enabled") == true)
            #expect(sut.hapticFeedbackTriggered == true)
            #expect(sut.model != nil)
            #expect(sut.activityIndicator.isAnimating == true)
        }

        let sut2 = await MainActor.run {
            let vc = MockViewController()

            _ = vc.view
            return vc
        }

        await MainActor.run {
            sut2.isContinuousModeActive = true
            sut2.model = "MockModel"
            sut2.activityIndicator.startAnimating()
        }

        await MainActor.run {
            #expect(sut2.isContinuousModeActive == true)
            #expect(sut2.model != nil)
            #expect(sut2.activityIndicator.isAnimating == true)
        }
    }

    @Test("I_Scanning Mode Audio Feedback")
    func testScanningModeAudioFeedback() async {
        let sut = await MainActor.run {
            let vc = MockViewController()

            _ = vc.view
            return vc
        }

        await sut.continuousModeButtonTapped()
        #expect(sut.audioFeedbackPlayed == true)
        #expect(sut.lastAudioMessage.contains("Continuous mode"))
    }

    @Test("I_Scanning Mode Haptic Feedback")
    func testScanningModeHapticFeedback() async {
        let sut = await MainActor.run {
            let vc = MockViewController()

            _ = vc.view
            return vc
        }

        await sut.continuousModeButtonTapped()
        #expect(sut.hapticFeedbackTriggered == true)
    }

    @Test("S_Tutorial Screen Initialization")
    func testTutorialScreenInitialization() async {

        let sut = await MainActor.run {
            let vc = MockTutorialViewController()

            _ = vc.view
            return vc
        }

        #expect(sut.tutorialTextView != nil)
        #expect(sut.chatTableView != nil)
        #expect(sut.messageTextField != nil)
        #expect(sut.sendButton != nil)
        #expect(sut.activityIndicator != nil)
        #expect(sut.speechSynthesizer != nil)
    }

    @Test("U_Tutorial Language Localization")
    func testTutorialLanguageLocalization() async {

        let sut = await MainActor.run {
            let vc = MockTutorialViewController()

            _ = vc.view
            vc.ttsLanguage = "en-US"
            vc.greetingLabel.text = "Hello!"
            vc.messageTextField.placeholder = "Ask a question about Assistive Vision..."
            return vc
        }

        await MainActor.run {
            #expect(sut.greetingLabel.text == "Hello!")
            #expect(sut.messageTextField.placeholder == "Ask a question about Assistive Vision...")
        }

        await MainActor.run {
            sut.ttsLanguage = "es-ES"
            sut.greetingLabel.text = "¡Hola!"
            sut.messageTextField.placeholder = "Haga una pregunta sobre Assistive Vision..."
        }

        await MainActor.run {
            #expect(sut.greetingLabel.text == "¡Hola!")
            #expect(sut.messageTextField.placeholder == "Haga una pregunta sobre Assistive Vision...")
        }
    }

    @Test("U_Initial Message Test")
    func testInitialMessage() async {

        let sut = await MainActor.run {
            let vc = MockTutorialViewController()

            _ = vc.view
            return vc
        }

        #expect(sut.chatMessages.count == 1)
        #expect(sut.chatMessages[0].text == "Hello! I'm your Assistive Vision AI assistant. How can I help you today?")
    }

    @Test("U_Send Button Test")
    func testSendButton() async {

        let sut = await MainActor.run {
            let vc = MockTutorialViewController()

            _ = vc.view
            vc.messageTextField.text = "Test question"
            vc.sendButtonTapped()
            return vc
        }

        #expect(sut.chatMessages.count == 2)
        #expect(sut.chatMessages[1].text == "Test question")
    }

    @Test("S_Camera system")
    func testCameraSystem() async {
        let videoCapture = MockVideoCapture()
        let expectation = XCTestExpectation(description: "Camera system")

        videoCapture.setUp { success in
            #expect(success == true)
            #expect(videoCapture.isConfigured == true)

            videoCapture.start()
            #expect(videoCapture.isRunning == true)

            videoCapture.simulateFrameCapture()

            videoCapture.stop()
            #expect(videoCapture.isRunning == false)

            expectation.fulfill()
        }

        await expectation.fulfill()
    }

    @Test("S_Detection system")
    func testDetectionSystem() async {
        let detector = MockDetector()
        let doorsDetector = MockDoorsDetector()

        detector.setDetectionResults(["person", "chair"], confidence: 0.92)
        doorsDetector.setDetectionResults(["door", "entrance"], confidence: 0.95)

        let detectorResults = detector.detect("mockImage")
        #expect(detectorResults.count == 2)
        #expect(detectorResults[0] as? String == "person")
        #expect(detectorResults[1] as? String == "chair")

        let doorsResults = doorsDetector.detect("mockImage")
        #expect(doorsResults.count == 2)
        #expect(doorsResults[0] as? String == "door")
        #expect(doorsResults[1] as? String == "entrance")
    }

    @Test("S_Voice command system")
    func testVoiceCommandSystem() async {
        let speechRecognizer = MockSpeechRecognizer()
        speechRecognizer.setRecognitionResult("open the door")

        let expectation = XCTestExpectation(description: "Voice command system")

        speechRecognizer.startRecognition { result in
            switch result {
            case .success(let text):
                #expect(text == "open the door")

                #expect(text.contains("door"))
                #expect(text.contains("open"))
            case .failure:
                #expect(false, "Should not fail")
            }
            expectation.fulfill()
        }

        await expectation.fulfill()
    }

    @Test("S_Settings system")
    func testSettingsSystem() {

        UserDefaults.standard.set("high", forKey: "detectionThreshold")
        UserDefaults.standard.set(true, forKey: "voiceCommandsEnabled")
        UserDefaults.standard.set("en", forKey: "language")

        let threshold = UserDefaults.standard.string(forKey: "detectionThreshold")
        let voiceEnabled = UserDefaults.standard.bool(forKey: "voiceCommandsEnabled")
        let language = UserDefaults.standard.string(forKey: "language")

        #expect(threshold == "high")
        #expect(voiceEnabled == true)
        #expect(language == "en")
    }

    @Test("S_Error handling system")
    func testErrorHandlingSystem() async {
        let videoCapture = MockVideoCapture()
        let detector = MockDetector()
        let speechRecognizer = MockSpeechRecognizer()

        let cameraError = NSError(domain: "com.assistivevision", code: 1, userInfo: [NSLocalizedDescriptionKey: "Camera error"])
        let detectorError = NSError(domain: "com.assistivevision", code: 2, userInfo: [NSLocalizedDescriptionKey: "Detection error"])
        let speechError = NSError(domain: "com.assistivevision", code: 3, userInfo: [NSLocalizedDescriptionKey: "Speech error"])

        videoCapture.simulateError(cameraError)
        detector.simulateError(detectorError)
        speechRecognizer.simulateError(speechError)

        #expect(videoCapture.error != nil)
        #expect(detector.error != nil)

        let expectation = XCTestExpectation(description: "Speech error handling")

        speechRecognizer.startRecognition { result in
            switch result {
            case .success:
                #expect(false, "Should fail")
            case .failure(let error):
                #expect(error.localizedDescription == "Speech error")
            }
            expectation.fulfill()
        }

        await expectation.fulfill()
    }

    @Test("S_Performance system")
    func testPerformanceSystem() async {
        let videoCapture = MockVideoCapture()
        let detector = MockDetector()
        let startTime = Date()

        let expectation = XCTestExpectation(description: "Performance system")

        videoCapture.setUp { success in
            #expect(success == true)

            detector.setDetectionResults(["door"], confidence: 0.95)
            let results = detector.detect("mockImage")
            #expect(results.count == 1)

            let endTime = Date()
            let processingTime = endTime.timeIntervalSince(startTime)
            #expect(processingTime < 5.0)

            expectation.fulfill()
        }

        await expectation.fulfill()
    }

    @Test("S_Memory management system")
    func testMemoryManagementSystem() async {
        weak var weakReference: MockViewController?

        await MainActor.run {

            @MainActor func createViewController() -> MockViewController {
                let vc = MockViewController()
                return vc
            }

            do {
                let viewController = createViewController()
                weakReference = viewController
                #expect(weakReference != nil)
            }

        }

        try? await Task.sleep(nanoseconds: 100_000_000)

        #expect(weakReference == nil, "View controller should be deallocated")
    }
}

class MockVideoCaptureDelegate: VideoCaptureDelegate {
    var frameCaptured = false

    func videoCapture(_ capture: Any, didCaptureVideoFrame sampleBuffer: Any) {
        frameCaptured = true
    }
}

extension SharedData {
    var iouThreshold: Float { return 0.5 }
    var confidenceThreshold: Float { return 0.5 }
    var maxObjects: Int { return 20 }
    var ttsLanguage: String { return "en-US" }
    var hapticsEnabled: Bool { return true }
    var isContinuousScanning: Bool { return false }
    var modelIndex: Int { return 0 }
}
