import 'package:flutter/material.dart';

import '../theme.dart';
import 'video_background.dart';

/// Shared visual frame for every screen opened from the Owner dashboard.
class OwnerFeatureFrame extends StatelessWidget {
  final Widget child;
  final double overlayOpacity;

  const OwnerFeatureFrame({
    super.key,
    required this.child,
    this.overlayOpacity = 0.48,
  });

  @override
  Widget build(BuildContext context) {
    return VideoBackground(
      assetPath: 'assets/images/background.mp4',
      overlayOpacity: overlayOpacity,
      tintColor: AppTheme.tealDeep,
      tintOpacity: 0.34,
      cardMargin: const EdgeInsets.all(14),
      cardRadius: 32,
      child: child,
    );
  }
}

abstract final class OwnerFeaturePalette {
  static const surface = Color(0xE600383A);
  static const surfaceSoft = Color(0xD9033032);
  static const surfaceStrong = Color(0xF003292B);
  static const border = Color(0x4D8AD7BF);
  static const text = Colors.white;
  static const textMuted = Color(0xBFD0E8E5);
}
