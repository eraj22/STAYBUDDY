import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../routes.dart';
import '../theme.dart';
import '../theme/responsive.dart';
import '../widgets/video_scaffold.dart';
import '../widgets/three_d_container.dart';
import '../widgets/app_button.dart';

class WelcomeScreen extends StatefulWidget {
  const WelcomeScreen({super.key});

  @override
  State<WelcomeScreen> createState() => _WelcomeScreenState();
}

class _WelcomeScreenState extends State<WelcomeScreen>
    with SingleTickerProviderStateMixin {
  Timer? _timer;

  late final AnimationController _controller;
  late final Animation<double> _bgFade;
  late final Animation<double> _bgScale;
  late final Animation<double> _cardSlide;
  late final Animation<double> _cardFade;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    );

    _bgFade = CurvedAnimation(parent: _controller, curve: Curves.easeOut);
    _bgScale = Tween<double>(begin: 1.04, end: 1.0).animate(
      CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic),
    );
    _cardFade = CurvedAnimation(
      parent: _controller,
      curve: const Interval(0.3, 1.0, curve: Curves.easeOut),
    );
    _cardSlide = Tween<double>(begin: 80, end: 0).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.3, 1.0, curve: Curves.easeOutCubic),
      ),
    );

    _controller.forward();

    _timer = Timer(const Duration(seconds: 10), () {
      if (!mounted) return;
      Navigator.pushReplacementNamed(context, Routes.role);
    });
  }

  @override
  void dispose() {
    _timer?.cancel();
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return VideoScaffold(
      child: FadeTransition(
        opacity: _bgFade,
        child: ScaleTransition(
          scale: _bgScale,
          child: SafeArea(
            child: Column(
              children: [
                // Top illustration area — flexes to fill whatever space
                // is left above the bottom card, on any screen height.
                Expanded(
                  child: Stack(
                    alignment: Alignment.center,
                    children: [
                      Positioned(
                        top: -40,
                        right: -40,
                        child: Container(
                          width: context.wp(180),
                          height: context.wp(180),
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.white.withOpacity(0.04),
                          ),
                        ),
                      ),
                      Positioned(
                        bottom: 30,
                        left: -30,
                        child: Container(
                          width: context.wp(120),
                          height: context.wp(120),
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: Colors.white.withOpacity(0.04),
                          ),
                        ),
                      ),
                      _buildCityIllustration(context),
                    ],
                  ),
                ),

                // Bottom card — height driven by content + safe padding,
                // never a hardcoded pixel value, so it never overflows
                // on short devices (iPhone SE, small Android phones).
                AnimatedBuilder(
                  animation: _controller,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, _cardSlide.value),
                    child: Opacity(opacity: _cardFade.value, child: child),
                  ),
                  child: _buildBottomCard(context),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildCityIllustration(BuildContext context) {
    final short = context.isShortScreen;
    return Padding(
      padding: EdgeInsets.symmetric(horizontal: context.wp(32)),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: EdgeInsets.symmetric(
              horizontal: context.wp(12),
              vertical: context.hp(5),
            ),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.12),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: Colors.white.withOpacity(0.2)),
            ),
            child: Text(
              'Student Housing',
              style: GoogleFonts.dmSans(
                fontSize: context.sp(11),
                color: Colors.white.withOpacity(0.8),
                fontWeight: FontWeight.w500,
                letterSpacing: 1.0,
              ),
            ),
          ),
          SizedBox(height: short ? context.hp(16) : context.hp(28)),
          FittedBox(
            fit: BoxFit.scaleDown,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                _buildBuilding(context, width: 44, height: 80, opacity: 0.15, windows: 3),
                SizedBox(width: context.wp(6)),
                _buildBuilding(context, width: 58, height: 120, opacity: 0.28, windows: 5),
                SizedBox(width: context.wp(6)),
                _buildBuilding(context, width: 46, height: 70, opacity: 0.20, windows: 2, accent: true),
                SizedBox(width: context.wp(6)),
                _buildBuilding(context, width: 60, height: 100, opacity: 0.25, windows: 4),
                SizedBox(width: context.wp(6)),
                _buildBuilding(context, width: 38, height: 60, opacity: 0.14, windows: 2),
              ],
            ),
          ),
          Container(
            height: 2,
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.2),
              borderRadius: BorderRadius.circular(1),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBuilding(
    BuildContext context, {
    required double width,
    required double height,
    required double opacity,
    required int windows,
    bool accent = false,
  }) {
    return Container(
      width: context.wp(width),
      height: context.hp(height),
      decoration: BoxDecoration(
        color: accent
            ? Colors.white.withOpacity(opacity + 0.08)
            : Colors.white.withOpacity(opacity),
        borderRadius: const BorderRadius.vertical(top: Radius.circular(4)),
        border: accent
            ? Border.all(color: AppTheme.tealMint.withOpacity(0.4), width: 1.5)
            : null,
      ),
      child: Padding(
        padding: const EdgeInsets.all(5),
        child: Column(
          children: List.generate(
            windows,
            (i) => Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  _buildWindow(accent),
                  _buildWindow(accent),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildWindow(bool accent) {
    return Container(
      width: 8,
      height: 10,
      decoration: BoxDecoration(
        color: accent
            ? AppTheme.tealMint.withOpacity(0.6)
            : Colors.white.withOpacity(0.4),
        borderRadius: BorderRadius.circular(2),
      ),
    );
  }

  Widget _buildBottomCard(BuildContext context) {
    final short = context.isShortScreen;
    return ThreeDContainer(
      borderRadius: 30.0,
      child: Container(
        width: double.infinity,
        constraints: BoxConstraints(
          maxHeight: MediaQuery.of(context).size.height * 0.62,
        ),
        decoration: const BoxDecoration(
          color: AppTheme.bg,
          borderRadius: BorderRadius.vertical(top: Radius.circular(30)),
        ),
        child: SingleChildScrollView(
          padding: EdgeInsets.fromLTRB(
            context.wp(28),
            context.hp(28),
            context.wp(28),
            context.hp(36),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  margin: EdgeInsets.only(bottom: short ? context.hp(14) : context.hp(22)),
                  decoration: BoxDecoration(
                    color: AppTheme.border,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),

              Text(
                'Find your\nperfect stay',
                style: GoogleFonts.playfairDisplay(
                  fontSize: context.sp(short ? 24 : 28),
                  fontWeight: FontWeight.w700,
                  color: AppTheme.textPrimary,
                  height: 1.15,
                ),
              ),
              SizedBox(height: context.hp(10)),
              Text(
                'Discover verified hostels near your university with real-time availability and AI-powered recommendations.',
                style: GoogleFonts.dmSans(
                  fontSize: context.sp(13),
                  color: AppTheme.textMuted,
                  height: 1.5,
                ),
              ),
              SizedBox(height: short ? context.hp(16) : context.hp(28)),

              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: const [
                  _FeaturePill(icon: Icons.verified_rounded, label: 'Verified'),
                  _FeaturePill(icon: Icons.location_on_rounded, label: 'Nearby'),
                  _FeaturePill(icon: Icons.star_rounded, label: 'AI Picks'),
                ],
              ),
              SizedBox(height: short ? context.hp(16) : context.hp(28)),

              AppPrimaryButton(
                label: 'Get started',
                trailingIcon: Icons.arrow_forward_rounded,
                onPressed: () =>
                    Navigator.pushReplacementNamed(context, Routes.role),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _FeaturePill extends StatelessWidget {
  final IconData icon;
  final String label;
  const _FeaturePill({required this.icon, required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: AppTheme.teal.withOpacity(0.08),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.teal.withOpacity(0.2)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: AppTheme.teal),
          const SizedBox(width: 5),
          Text(
            label,
            style: GoogleFonts.dmSans(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: AppTheme.teal,
            ),
          ),
        ],
      ),
    );
  }
}