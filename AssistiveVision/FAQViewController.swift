import UIKit
import AVFoundation

class FAQViewController: UIViewController {
    @IBOutlet weak var faqTextView: UITextView!
    var shouldReadText = false
    let speechSynthesizer = AVSpeechSynthesizer()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        if SharedData.shared.ttsLanguage == "es-ES" {
            faqTextView.text = """
            Preguntas Frecuentes

            1. ¿Qué es Assistive Vision?
            Assistive Vision es una aplicación para iOS que ayuda a personas con discapacidad visual a navegar su entorno. Utiliza un modelo YOLOv11m personalizado con CoreML y el Neural Engine de Apple para detectar objetos en tiempo real y proporcionar retroalimentación auditiva inmediata.

            2. ¿Cuál es el propósito y alcance del proyecto?
            El objetivo es aumentar la independencia de los usuarios permitiéndoles identificar y localizar objetos y obstáculos en su entorno sin asistencia externa. La app ofrece modos de escaneo continuo y bajo demanda, ajustes personalizables y una experiencia totalmente accesible.

            3. ¿Qué modos de escaneo están disponibles?
            La aplicación permite dos modos:
            - Escaneo Continuo: Detecta objetos en tiempo real de forma constante.
            - Detección Bajo Demanda: Realiza detección solo cuando el usuario lo solicita.

            4. ¿Cómo se inicia y cierra la aplicación?
            Los usuarios pueden lanzar Assistive Vision tocando el ícono de la app. La app inicia la cámara, el modelo de detección y el motor de audio. Para salir, se pueden utilizar gestos estándar de iOS.

            5. ¿Cómo se personalizan los ajustes de detección?
            Los usuarios pueden ajustar umbrales de confianza, priorizar ciertas categorías de objetos y configurar la retroalimentación en audio y vibratoria desde el menú de ajustes.

            6. ¿Cómo se protege la privacidad del usuario?
            Toda la detección se realiza localmente en el dispositivo sin enviar datos a la nube. Esto garantiza que las imágenes y la información personal permanezcan privadas.

            7. ¿Qué funciones de accesibilidad están integradas?
            La aplicación se integra con VoiceOver y otras herramientas de accesibilidad de iOS, ofreciendo una experiencia completamente adaptada a las necesidades de los usuarios con discapacidad visual.

            8. ¿Cómo se gestiona el consumo de batería?
            Dado que el escaneo continuo puede consumir mucha batería, la app incorpora un Modo Ahorro de Batería que reduce la frecuencia de procesamiento para prolongar la vida útil del dispositivo.

            9. ¿Qué debo hacer si encuentro un problema?
            Si se presentan inconvenientes, la app proporcionará mensajes claros (por ejemplo, alertas de permisos o problemas de hardware) y sugerirá pasos para solucionarlos, como revisar la configuración del dispositivo o reiniciar la aplicación.

            10. ¿Qué mejoras se esperan para el futuro?
            Se planea expandir las capacidades de detección, mejorar la personalización de la retroalimentación y optimizar la experiencia para hacerla aún más fluida y accesible.
            """
        } else {
            faqTextView.text = """
            Frequently Asked Questions

            1. What is Assistive Vision?
            Assistive Vision is an iOS app designed to help visually impaired users navigate their surroundings. It employs a custom YOLOv11m model using CoreML and Apple's Neural Engine to detect objects in real time, providing immediate audio feedback.

            2. What is the purpose and scope of the project?
            The project aims to empower users by enabling them to identify and locate objects and obstacles independently. The app offers both continuous and on-demand scanning modes, customizable detection settings, and a fully accessible user experience.

            3. What scanning modes are available?
            The app provides two modes:
            - Continuous Scanning: Constant real-time detection.
            - On-Demand Detection: Scans only when the user requests it.

            4. How do I launch and exit the app?
            Users can launch Assistive Vision by tapping the app icon. On launch, the app initializes the camera, detection model, and audio engine. To exit, use standard iOS gestures.

            5. How are detection settings customized?
            Users can adjust confidence thresholds, prioritize specific object categories, and configure audio and haptic feedback via the settings menu. Changes take effect immediately.

            6. How is user privacy maintained?
            All object detection and processing occur on-device, with no data transmitted to external servers. This ensures that personal images and information remain private.

            7. What accessibility features does the app support?
            The app integrates with iOS accessibility tools such as VoiceOver, enabling a fully intuitive experience tailored to visually impaired users.

            8. How does the app manage battery usage?
            To conserve battery during continuous scanning, the app features a Battery Saver Mode that reduces frame processing and detection frequency when needed.

            9. What should I do if I encounter a problem?
            The app provides clear audio prompts for troubleshooting issues, such as permission errors or hardware problems, and offers guidance on resolving them.

            10. What future enhancements are planned?
            Future updates may include expanded object detection capabilities, improved navigation features, and more personalized audio feedback.
            """
        }
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        if shouldReadText {
            DispatchQueue.main.asyncAfter(deadline: .now() + 5.5) {
                self.speakText(self.faqTextView.text)
            }
            shouldReadText = false
        }
    }
    
    @IBAction func readButtonTapped(_ sender: UIButton) {
        speakText(faqTextView.text)
    }
    
    func speakText(_ text: String) {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .default, options: .duckOthers)
            try audioSession.overrideOutputAudioPort(.speaker)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print(error)
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: SharedData.shared.ttsLanguage)
        speechSynthesizer.speak(utterance)
    }
    
    func speakMessage(_ message: String) {
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playAndRecord, mode: .default, options: .duckOthers)
            try audioSession.overrideOutputAudioPort(.speaker)
            try audioSession.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print(error)
        }
        let utterance = AVSpeechUtterance(string: message)
        utterance.voice = AVSpeechSynthesisVoice(language: SharedData.shared.ttsLanguage)
        speechSynthesizer.speak(utterance)
    }
    
    func l(_ en: String, _ es: String) -> String {
        return SharedData.shared.ttsLanguage == "es-ES" ? es : en
    }
}
