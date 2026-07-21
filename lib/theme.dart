import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class AppTheme {
  // Primary teal palette
  static const teal = Color(0xFF0A6B6E);
  static const tealDark = Color(0xFF074F51);
  static const tealDeep = Color(0xFF063E40);
  static const tealLight = Color(0xFF1D9E75);
  static const tealMint = Color(0xFF5DCAA5);

  // Neutral warm palette
  static const bg = Color(0xFFF5F3EE);         // warm cream
  static const bgSecondary = Color(0xFFECEAE4); // slightly darker cream
  static const card = Color(0xFFFFFFFF);
  static const border = Color(0xFFE0DDD6);
  static const borderLight = Color(0xFFD3D1C7);

  // Text
  static const textPrimary = Color(0xFF0A3D3F);
  static const textSecondary = Color(0xFF5F5E5A);
  static const textMuted = Color(0xFF888780);

  // Accent
  static const amber = Color(0xFFEF9F27);
  static const green = Color(0xFF3B6D11);
  static const greenLight = Color(0xFFEAF3DE);

  static ThemeData light() {
    final base = ThemeData(
      useMaterial3: true,
      scaffoldBackgroundColor: bg,
      colorScheme: ColorScheme.fromSeed(
        seedColor: teal,
        primary: teal,
        background: bg,
        surface: card,
      ),
    );

    return base.copyWith(
      textTheme: GoogleFonts.dmSansTextTheme(base.textTheme).copyWith(
        displayLarge: GoogleFonts.playfairDisplay(
          fontSize: 34, fontWeight: FontWeight.w700, color: textPrimary,
        ),
        displayMedium: GoogleFonts.playfairDisplay(
          fontSize: 28, fontWeight: FontWeight.w700, color: textPrimary,
        ),
        displaySmall: GoogleFonts.playfairDisplay(
          fontSize: 22, fontWeight: FontWeight.w600, color: textPrimary,
        ),
        headlineMedium: GoogleFonts.playfairDisplay(
          fontSize: 20, fontWeight: FontWeight.w600, color: textPrimary,
        ),
        headlineSmall: GoogleFonts.dmSans(
          fontSize: 17, fontWeight: FontWeight.w700, color: textPrimary,
        ),
        titleLarge: GoogleFonts.dmSans(
          fontSize: 16, fontWeight: FontWeight.w600, color: textPrimary,
        ),
        bodyLarge: GoogleFonts.dmSans(
          fontSize: 15, fontWeight: FontWeight.w400, color: textPrimary,
        ),
        bodyMedium: GoogleFonts.dmSans(
          fontSize: 13, fontWeight: FontWeight.w400, color: textSecondary,
        ),
        labelSmall: GoogleFonts.dmSans(
          fontSize: 11, fontWeight: FontWeight.w500, color: textMuted,
          letterSpacing: 0.5,
        ),
      ),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: card,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide(color: border),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide(color: border),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: const BorderSide(color: teal, width: 1.5),
        ),
        hintStyle: GoogleFonts.dmSans(
          color: textMuted, fontSize: 14,
        ),
        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        prefixIconColor: textMuted,
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: teal,
          foregroundColor: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          textStyle: GoogleFonts.dmSans(
            fontSize: 15, fontWeight: FontWeight.w600,
          ),
          padding: const EdgeInsets.symmetric(vertical: 14),
        ),
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: teal,
        foregroundColor: Colors.white,
        elevation: 0,
        centerTitle: false,
        titleTextStyle: GoogleFonts.dmSans(
          fontSize: 17, fontWeight: FontWeight.w600, color: Colors.white,
        ),
      ),
      chipTheme: ChipThemeData(
        backgroundColor: bgSecondary,
        labelStyle: GoogleFonts.dmSans(fontSize: 12, color: textSecondary),
        side: BorderSide(color: border),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      ),
    );
  }
}