import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../theme/app_sizes.dart';

/// Full-width primary action button — always 56dp tall, always the same
/// radius and text style, across every screen.
class AppPrimaryButton extends StatelessWidget {
  final String label;
  final VoidCallback? onPressed;
  final IconData? trailingIcon;
  final bool loading;
  final Color? backgroundColor;

  const AppPrimaryButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.trailingIcon,
    this.loading = false,
    this.backgroundColor,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: AppSizes.buttonHeight,
      child: ElevatedButton(
        onPressed: loading ? null : onPressed,
        style: ElevatedButton.styleFrom(
          backgroundColor: backgroundColor ?? AppTheme.teal,
          foregroundColor: Colors.white,
          disabledBackgroundColor: (backgroundColor ?? AppTheme.teal).withOpacity(0.5),
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
          ),
        ),
        child: loading
            ? const SizedBox(
                width: 22,
                height: 22,
                child: CircularProgressIndicator(
                  strokeWidth: 2.4,
                  valueColor: AlwaysStoppedAnimation(Colors.white),
                ),
              )
            : Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text(
                    label,
                    style: GoogleFonts.dmSans(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  if (trailingIcon != null) ...[
                    const SizedBox(width: 10),
                    Icon(trailingIcon, size: 18),
                  ],
                ],
              ),
      ),
    );
  }
}

/// Full-width outlined/secondary button — same 56dp height as primary so
/// the two never look mismatched when stacked.
class AppSecondaryButton extends StatelessWidget {
  final String label;
  final VoidCallback? onPressed;

  const AppSecondaryButton({super.key, required this.label, required this.onPressed});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      height: AppSizes.buttonHeight,
      child: OutlinedButton(
        onPressed: onPressed,
        style: OutlinedButton.styleFrom(
          foregroundColor: AppTheme.teal,
          side: const BorderSide(color: AppTheme.teal, width: 1.4),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(AppSizes.radiusMedium),
          ),
        ),
        child: Text(
          label,
          style: GoogleFonts.dmSans(fontSize: 16, fontWeight: FontWeight.w600),
        ),
      ),
    );
  }
}

/// Circular icon-only button — always meets the 48dp minimum touch target.
/// Use for "next" arrows, back buttons on video backgrounds, etc.
class AppCircleButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback? onPressed;
  final double size;
  final Color? backgroundColor;
  final Color? iconColor;
  final bool useGradient;

  const AppCircleButton({
    super.key,
    required this.icon,
    required this.onPressed,
    this.size = AppSizes.iconButtonSize,
    this.backgroundColor,
    this.iconColor,
    this.useGradient = false,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onPressed,
      child: Container(
        width: size < AppSizes.minTouchTarget ? AppSizes.minTouchTarget : size,
        height: size < AppSizes.minTouchTarget ? AppSizes.minTouchTarget : size,
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          gradient: useGradient
              ? const LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [AppTheme.tealLight, AppTheme.teal],
                )
              : null,
          color: useGradient ? null : (backgroundColor ?? Colors.white.withOpacity(0.16)),
          boxShadow: useGradient
              ? [
                  BoxShadow(
                    color: AppTheme.teal.withOpacity(0.45),
                    blurRadius: 18,
                    offset: const Offset(0, 8),
                  ),
                ]
              : null,
        ),
        child: Icon(icon, color: iconColor ?? Colors.white, size: size * 0.42),
      ),
    );
  }
}