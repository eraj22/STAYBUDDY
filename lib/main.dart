import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'theme.dart';
import 'routes.dart';
import 'firebase_options.dart';

import 'screens/splash_screen.dart';
import 'screens/welcome_screen.dart';
import 'screens/role_screen.dart';
import 'screens/login_screen.dart';
import 'screens/signup_screen.dart';
import 'screens/hostels_screen.dart';
import 'screens/discover_hostel_home_page.dart';
import 'screens/location_access_screen.dart';
import 'screens/owner_login_screen.dart';
import 'screens/owner_info_screen.dart';
import 'screens/ai_recommendation_screen.dart';
import 'widgets/responsive_shell.dart';
import 'services/app_language.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  await AppLanguage.load();

  runApp(const StayBuddyStudentApp());
}

class StayBuddyStudentApp extends StatelessWidget {
  const StayBuddyStudentApp({super.key});

  @override
  Widget build(BuildContext context) {
    final appRoutes = <String, WidgetBuilder>{
      Routes.splash: (_) => const SplashScreen(),
      Routes.welcome: (_) => const WelcomeScreen(),
      Routes.role: (_) => const RoleScreen(),
      Routes.login: (_) => const LoginScreen(),
      Routes.signup: (_) => const SignupScreen(),
      Routes.hostels: (_) => const HostelsScreen(),
      Routes.discoverHostelHome: (_) => const DiscoverHostelHomePage(),
      Routes.locationAccess: (_) => const LocationAccessScreen(),
      Routes.aiRecommendation: (_) => const AiRecommendationScreen(),
      Routes.ownerLogin: (_) => const OwnerLoginScreen(),
      Routes.ownerInfo: (_) => const OwnerInfoScreen(),
    };

    return ValueListenableBuilder<Locale>(
      valueListenable: AppLanguage.locale,
      builder: (context, locale, _) => MaterialApp(
      title: "StayBuddy",
      locale: locale,
      builder: (context, child) => Directionality(
        textDirection: locale.languageCode == 'ur'
            ? TextDirection.rtl
            : TextDirection.ltr,
        child: child ?? const SizedBox.shrink(),
      ),
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light(),
      // The app Navigator and ScaffoldMessenger live inside the phone frame.
      // Their overlays (dialogs, sheets, menus and SnackBars) therefore can
      // never expand to the surrounding desktop browser canvas.
      home: ResponsiveShell(
        child: ScaffoldMessenger(
          child: Navigator(
            initialRoute: Routes.splash,
            onGenerateRoute: (settings) {
              final routeBuilder = appRoutes[settings.name];
              if (routeBuilder == null) return null;
              return MaterialPageRoute<void>(
                settings: settings,
                builder: routeBuilder,
              );
            },
          ),
        ),
      ),
      ),
    );
  }
}
