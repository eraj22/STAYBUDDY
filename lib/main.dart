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
import 'screens/ai_room_search_screen.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  runApp(const StayBuddyStudentApp());
}

class StayBuddyStudentApp extends StatelessWidget {
  const StayBuddyStudentApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "StayBuddy",
      debugShowCheckedModeBanner: false,
      theme: AppTheme.light(),
      initialRoute: Routes.splash,
      routes: {
        Routes.splash: (_) => const SplashScreen(),
        Routes.welcome: (_) => const WelcomeScreen(),
        Routes.role: (_) => const RoleScreen(),
        Routes.login: (_) => const LoginScreen(),
        Routes.signup: (_) => const SignupScreen(),
        Routes.hostels: (_) => const HostelsScreen(),
        Routes.discoverHostelHome: (_) => const DiscoverHostelHomePage(),
        Routes.locationAccess: (_) => const LocationAccessScreen(),
        // AI features
        Routes.aiRecommendation: (_) => const AiRecommendationScreen(),
        // Owner flow
        Routes.ownerLogin: (_) => const OwnerLoginScreen(),
        Routes.ownerInfo: (_) => const OwnerInfoScreen(),
      },
    );
  }
}