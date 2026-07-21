import 'package:flutter/material.dart';

/// Scales fixed design values (padding, font size, spacing) proportionally
/// to the ACTUAL device width/height, instead of using the same raw pixel
/// number on every phone regardless of screen size.
///
/// Reference design size = 390 x 844 (standard modern phone, e.g. iPhone
/// 13/14). All screens were designed against that size — these helpers
/// scale everything up/down for smaller or larger real devices.
extension ResponsiveSizing on BuildContext {
  double get _screenWidth => MediaQuery.of(this).size.width;
  double get _screenHeight => MediaQuery.of(this).size.height;

  /// Scale a width-based value (padding, horizontal spacing, card widths).
  double wp(double designPx) {
    final scale = _screenWidth / 390.0;
    // Clamp so things never shrink/grow to an unusable extreme on very
    // small or very large devices.
    return designPx * scale.clamp(0.85, 1.25);
  }

  /// Scale a height-based value (vertical spacing, section heights).
  double hp(double designPx) {
    final scale = _screenHeight / 844.0;
    return designPx * scale.clamp(0.80, 1.20);
  }

  /// Scale a font size — uses width scale but clamps tighter so text
  /// never becomes too large or too small to read comfortably.
  double sp(double designPx) {
    final scale = _screenWidth / 390.0;
    return designPx * scale.clamp(0.90, 1.15);
  }

  /// True on short devices (e.g. iPhone SE, small Android phones) where
  /// vertical space is tight — use to trim optional spacing/elements.
  bool get isShortScreen => _screenHeight < 700;
}