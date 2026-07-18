/// Single source of truth for sizing across the app.
/// Every screen should pull dimensions from here instead of hardcoding
/// pixel values — this is what keeps buttons, spacing, and touch targets
/// consistent across all 34 screens.
class AppSizes {
  AppSizes._();

  // Buttons
  static const double buttonHeight = 56.0;       // primary full-width buttons
  static const double buttonHeightCompact = 48.0; // secondary/inline buttons
  static const double iconButtonSize = 48.0;      // circular icon buttons (min touch target)
  static const double fabSize = 60.0;             // large floating action buttons

  // Touch targets — Apple HIG / Material both recommend >= 44-48dp minimum
  static const double minTouchTarget = 48.0;

  // Border radius scale
  static const double radiusSmall = 12.0;
  static const double radiusMedium = 16.0;
  static const double radiusLarge = 22.0;
  static const double radiusXLarge = 28.0;

  // Spacing scale (use multiples of 4)
  static const double space4 = 4.0;
  static const double space8 = 8.0;
  static const double space12 = 12.0;
  static const double space16 = 16.0;
  static const double space20 = 20.0;
  static const double space24 = 24.0;
  static const double space32 = 32.0;
  static const double space40 = 40.0;

  // Standard screen edge padding (mobile-safe, works down to 320dp width)
  static const double screenPadding = 20.0;

  // Avatars / icon circles
  static const double avatarSmall = 40.0;
  static const double avatarMedium = 48.0;
  static const double avatarLarge = 64.0;

  // Input fields
  static const double inputHeight = 56.0;
}