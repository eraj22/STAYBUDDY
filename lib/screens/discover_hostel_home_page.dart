import 'package:flutter/material.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import '../widgets/three_d_container.dart';
import '../routes.dart';

class DiscoverHostelHomePage extends StatelessWidget {
  const DiscoverHostelHomePage({super.key});

  @override
  Widget build(BuildContext context) {
    const teal = Color(0xFF0B7C80);

    return ResponsiveShell(
      child: Scaffold(
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.35,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 22),
            child: Column(
              children: [
                const SizedBox(height: 20), // ✅ reduced so logo can come down

                Expanded(
                  flex: 6,
                  child: Center(
                    child: Padding(
                      padding: const EdgeInsets.only(top: 35), // ✅ moves logo down
                      child: Image.asset(
                        "assets/images/logo.png",
                        width: 520,
                        fit: BoxFit.contain,
                      ),
                    ),
                  ),
                ),

                Expanded(
                  flex: 4,
                  child: Padding(
                    padding: const EdgeInsets.only(bottom: 25), // ✅ moves text/button up
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center, // ✅ not stuck at bottom
                      children: [
                        const Text(
                          "Want to discover best hostels?",
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            color: teal,
                            fontSize: 22,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        const SizedBox(height: 18),

                        ThreeDContainer(
                          borderRadius: 34,
                          child: SizedBox(
                            height: 58,
                            child: ElevatedButton(
                              style: ElevatedButton.styleFrom(
                                backgroundColor: teal,
                                foregroundColor: Colors.white,
                                elevation: 0,
                                padding: const EdgeInsets.symmetric(horizontal: 34),
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(34),
                                ),
                              ),
                              onPressed: () {
                                Navigator.pushReplacementNamed(
                                  context,
                                  Routes.locationAccess,
                                );
                              },
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: const [
                                  Text(
                                    "Start",
                                    style: TextStyle(
                                      fontSize: 22,
                                      fontWeight: FontWeight.w800,
                                    ),
                                  ),
                                  SizedBox(width: 16),
                                  CircleAvatar(
                                    radius: 16,
                                    backgroundColor: Colors.white,
                                    child: Icon(
                                      Icons.arrow_forward,
                                      size: 20,
                                      color: teal,
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 20),
              ],
            ),
          ),
        ),
      ),
    );
  }
}