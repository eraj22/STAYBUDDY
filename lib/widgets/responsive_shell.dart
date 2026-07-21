import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

/// Keeps the app at a real phone size everywhere.
///
/// - On an actual mobile device (not web): full screen, untouched.
/// - On web/desktop with a narrow window (< 700px, e.g. a phone browser):
///   full screen, treated exactly like mobile.
/// - On web/desktop with a wide window: the app is rendered inside a
///   fixed phone-sized frame (like an iPhone), centered on the page,
///   with the rest of the browser window letterboxed. This is what makes
///   it actually LOOK like a mobile app on a desktop browser instead of
///   stretching to fill the window.
class ResponsiveShell extends StatelessWidget {
  final Widget child;
  const ResponsiveShell({super.key, required this.child});

  // Standard modern phone dimensions (iPhone 14/15-ish).
  static const double _phoneWidth = 402.0;
  static const double _phoneHeight = 874.0;
  static const double _frameRadius = 40.0;

  @override
  Widget build(BuildContext context) {
    // On mobile (not web) — always full screen
    if (!kIsWeb) return child;

    final size = MediaQuery.of(context).size;

    // Web narrow (< 700px) — treat like mobile
    if (size.width < 700) return child;

    // Web wide — render inside a fixed phone-sized frame, centered,
    // with letterboxing around it. Never stretches to fill the window.
    final availableHeight = size.height - 64; // breathing room top/bottom
    final frameHeight =
        availableHeight < _phoneHeight ? availableHeight : _phoneHeight;
    final frameWidth = frameHeight * (_phoneWidth / _phoneHeight);

    return Scaffold(
      backgroundColor: const Color.fromARGB(255, 255, 255, 255),
      body: Center(
        child: Container(
          width: frameWidth,
          height: frameHeight,
          clipBehavior: Clip.antiAlias,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(_frameRadius),
            border: Border.all(color: Colors.black, width: 8),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.55),
                blurRadius: 50,
                spreadRadius: 2,
                offset: const Offset(0, 20),
              ),
            ],
          ),
          child: child,
        ),
      ),
    );
  }
}
