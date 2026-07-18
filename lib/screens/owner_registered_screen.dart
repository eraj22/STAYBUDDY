import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_hostel_info_screen.dart';

class OwnerRegisteredScreen extends StatefulWidget {
  final String name, email;
  const OwnerRegisteredScreen({super.key, required this.name, required this.email});

  @override
  State<OwnerRegisteredScreen> createState() => _OwnerRegisteredScreenState();
}

class _OwnerRegisteredScreenState extends State<OwnerRegisteredScreen>
    with TickerProviderStateMixin {
  late final AnimationController _checkAnim;
  late final AnimationController _textAnim;
  late final Animation<double> _checkScale;
  late final Animation<double> _checkOpacity;
  late final Animation<double> _ringExpand;
  late final Animation<double> _textFade;
  late final Animation<double> _textSlide;

  @override
  void initState() {
    super.initState();
    _checkAnim = AnimationController(vsync: this, duration: const Duration(milliseconds: 900));
    _textAnim = AnimationController(vsync: this, duration: const Duration(milliseconds: 600));

    _checkScale = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _checkAnim, curve: const Interval(0.2, 0.8, curve: Curves.elasticOut)));
    _checkOpacity = CurvedAnimation(parent: _checkAnim, curve: const Interval(0.0, 0.3, curve: Curves.easeOut));
    _ringExpand = Tween<double>(begin: 0.6, end: 1.0).animate(
      CurvedAnimation(parent: _checkAnim, curve: const Interval(0.0, 0.5, curve: Curves.easeOut)));
    _textFade = CurvedAnimation(parent: _textAnim, curve: Curves.easeOut);
    _textSlide = Tween<double>(begin: 20, end: 0).animate(
      CurvedAnimation(parent: _textAnim, curve: Curves.easeOutCubic));

    _checkAnim.forward().then((_) => _textAnim.forward());
  }

  @override
  void dispose() {
    _checkAnim.dispose();
    _textAnim.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.30,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: Column(children: [
              const Spacer(),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.fromLTRB(28, 28, 28, 36),
                decoration: const BoxDecoration(
                  color: AppTheme.bg,
                  borderRadius: BorderRadius.vertical(top: Radius.circular(32)),
                ),
                child: Column(children: [
                  Center(child: Container(width: 40, height: 4, margin: const EdgeInsets.only(bottom: 28),
                    decoration: BoxDecoration(color: AppTheme.border, borderRadius: BorderRadius.circular(2)))),

                  // Animated success icon
                  AnimatedBuilder(
                    animation: _checkAnim,
                    builder: (_, __) => ScaleTransition(
                      scale: _ringExpand,
                      child: FadeTransition(
                        opacity: _checkOpacity,
                        child: Stack(alignment: Alignment.center, children: [
                          // Outer glow ring
                          Container(width: 120, height: 120,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: const Color(0xFF4CAF50).withOpacity(0.1),
                            )),
                          // Middle ring
                          Container(width: 96, height: 96,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: const Color(0xFF4CAF50).withOpacity(0.15),
                            )),
                          // Check circle
                          ScaleTransition(
                            scale: _checkScale,
                            child: Container(width: 74, height: 74,
                              decoration: BoxDecoration(
                                color: const Color(0xFF4CAF50), shape: BoxShape.circle,
                                boxShadow: [BoxShadow(color: const Color(0xFF4CAF50).withOpacity(0.4), blurRadius: 20, spreadRadius: 2)],
                              ),
                              child: const Icon(Icons.check_rounded, color: Colors.white, size: 40)),
                          ),
                        ]),
                      ),
                    ),
                  ),
                  const SizedBox(height: 24),

                  AnimatedBuilder(
                    animation: _textAnim,
                    builder: (_, child) => Transform.translate(
                      offset: Offset(0, _textSlide.value),
                      child: Opacity(opacity: _textFade.value, child: child),
                    ),
                    child: Column(children: [
                      Text('Profile Created!', style: GoogleFonts.playfairDisplay(
                        fontSize: 26, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
                      const SizedBox(height: 8),
                      Text('Welcome, ${widget.name}!', style: GoogleFonts.dmSans(
                        fontSize: 15, fontWeight: FontWeight.w600, color: AppTheme.teal)),
                      const SizedBox(height: 6),
                      Text('Your owner account is ready.\nNow let\'s add your hostel details.',
                        textAlign: TextAlign.center,
                        style: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textMuted, height: 1.5)),
                      const SizedBox(height: 28),

                      // What's next card
                      Container(
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(color: AppTheme.teal.withOpacity(0.05),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: AppTheme.teal.withOpacity(0.15), width: 0.5)),
                        child: Column(children: [
                          Text("What's next?", style: GoogleFonts.dmSans(
                            fontSize: 13, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
                          const SizedBox(height: 10),
                          _nextItem(icon: Icons.business_rounded, label: 'Add hostel information'),
                          const SizedBox(height: 8),
                          _nextItem(icon: Icons.location_on_rounded, label: 'Pin hostel on map'),
                          const SizedBox(height: 8),
                          _nextItem(icon: Icons.bed_rounded, label: 'Set room & bed details'),
                        ]),
                      ),
                      const SizedBox(height: 24),

                      SizedBox(width: double.infinity, height: 52,
                        child: ElevatedButton(
                          onPressed: () => Navigator.pushReplacement(context, MaterialPageRoute(
                            builder: (_) => OwnerHostelInfoScreen(ownerEmail: widget.email, ownerName: widget.name))),
                          style: ElevatedButton.styleFrom(backgroundColor: AppTheme.teal, foregroundColor: Colors.white,
                            elevation: 0, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                          child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                            Text('Add My Hostel', style: GoogleFonts.dmSans(fontSize: 16, fontWeight: FontWeight.w600)),
                            const SizedBox(width: 10),
                            Container(width: 28, height: 28,
                              decoration: BoxDecoration(color: Colors.white.withOpacity(0.2), shape: BoxShape.circle),
                              child: const Icon(Icons.arrow_forward_rounded, size: 16)),
                          ]),
                        )),
                    ]),
                  ),
                ]),
              ),
            ]),
          ),
        ),
      ),
    );
  }

  Widget _nextItem({required IconData icon, required String label}) {
    return Row(children: [
      Container(width: 30, height: 30,
        decoration: BoxDecoration(color: AppTheme.teal.withOpacity(0.1), shape: BoxShape.circle),
        child: Icon(icon, size: 15, color: AppTheme.teal)),
      const SizedBox(width: 10),
      Text(label, style: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textSecondary)),
    ]);
  }
}