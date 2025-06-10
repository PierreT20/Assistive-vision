import UIKit

@UIApplicationMain
class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?
    
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        UIApplication.shared.isIdleTimerDisabled = true
        UIDevice.current.isBatteryMonitoringEnabled = true
        if let ver = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String,
           let build = Bundle.main.infoDictionary?["CFBundleVersion"] as? String {
            UserDefaults.standard.set("\(ver) (\(build))", forKey: "app_version")
        }
        if let devID = UIDevice.current.identifierForVendor?.uuidString {
            UserDefaults.standard.set(devID, forKey: "uuid")
        }
        UserDefaults.standard.synchronize()
        return true
    }
    
    func applicationWillTerminate(_ application: UIApplication) {
        print("Application will terminate")
    }
}
