import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_dashboard_screen.dart';

class OwnerHostelSuccessScreen extends StatefulWidget {
  final String ownerName, hostelName;
  const OwnerHostelSuccessScreen({super.key, required this.ownerName, required this.hostelName});

  @override
  State<OwnerHostelSuccessScreen> createState() => _OwnerHostelSuccessScreenState();
}

class _OwnerHostelSuccessScreenState extends State<OwnerHostelSuccessScreen>
    with TickerProviderStateMixin {
  late final AnimationController _iconAnim;
  late final AnimationController _contentAnim;

  late final Animation<double> _iconScale;
  late final Animation<double> _iconOpacity;
  late final Animation<double> _pulseAnim;
  late final Animation<double> _contentFade;
  late final Animation<double> _contentSlide;

  @override
  void initState() {
    super.initState();
    _iconAnim = AnimationController(vsync: this, duration: const Duration(milliseconds: 1000));
    _contentAnim = AnimationController(vsync: this, duration: const Duration(milliseconds: 600));

    _iconScale = Tween<double>(begin: 0, end: 1).animate(
      CurvedAnimation(parent: _iconAnim, curve: const Interval(0.1, 0.7, curve: Curves.elasticOut)));
    _iconOpacity = CurvedAnimation(parent: _iconAnim, curve: const Interval(0, 0.3, curve: Curves.easeOut));
    _pulseAnim = Tween<double>(begin: 1.0, end: 1.08).animate(
      CurvedAnimation(parent: _iconAnim, curve: const Interval(0.7, 1.0, curve: Curves.easeInOut)));

    _contentFade = CurvedAnimation(parent: _contentAnim, curve: Curves.easeOut);
    _contentSlide = Tween<double>(begin: 24, end: 0).animate(
      CurvedAnimation(parent: _contentAnim, curve: Curves.easeOutCubic));

    _iconAnim.forward().then((_) {
      _iconAnim.repeat(reverse: true);
      _contentAnim.forward();
    });
  }

  @override
  void dispose() {
    _iconAnim.dispose();
    _contentAnim.dispose();
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

                  // Animated map icon
                  AnimatedBuilder(
                    animation: _iconAnim,
                    builder: (_, __) => FadeTransition(
                      opacity: _iconOpacity,
                      child: ScaleTransition(
                        scale: _iconScale,
                        child: ScaleTransition(
                          scale: _pulseAnim,
                          child: Stack(alignment: Alignment.center, children: [
                            Container(width: 120, height: 120,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: AppTheme.teal.withOpacity(0.08))),
                            Container(width: 92, height: 92,
                              decoration: BoxDecoration(
                                shape: BoxShape.circle,
                                color: AppTheme.teal.withOpacity(0.12))),
                            Container(width: 70, height: 70,
                              decoration: BoxDecoration(
                                color: AppTheme.teal, shape: BoxShape.circle,
                                boxShadow: [BoxShadow(color: AppTheme.teal.withOpacity(0.4), blurRadius: 20, spreadRadius: 2)]),
                              child: const Icon(Icons.location_on_rounded, color: Colors.white, size: 36)),
                          ]),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 22),

                  AnimatedBuilder(
                    animation: _contentAnim,
                    builder: (_, child) => Transform.translate(
                      offset: Offset(0, _contentSlide.value),
                      child: Opacity(opacity: _contentFade.value, child: child)),
                    child: Column(children: [
                      Text('Hostel Listed!', style: GoogleFonts.playfairDisplay(
                        fontSize: 28, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
                      const SizedBox(height: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 7),
                        decoration: BoxDecoration(color: AppTheme.teal.withOpacity(0.08),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: AppTheme.teal.withOpacity(0.2))),
                        child: Text(widget.hostelName, style: GoogleFonts.dmSans(
                          fontSize: 14, fontWeight: FontWeight.w600, color: AppTheme.teal)),
                      ),
                      const SizedBox(height: 10),
                      Text('Your hostel is now live on the map.\nStudents can discover it right away!',
                        textAlign: TextAlign.center,
                        style: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textMuted, height: 1.55)),
                      const SizedBox(height: 24),

                      // Achievement cards
                      Row(children: [
                        Expanded(child: _achievementCard(
                          icon: Icons.map_rounded,
                          label: 'On Map',
                          sub: 'Visible to students',
                          color: AppTheme.teal)),
                        const SizedBox(width: 10),
                        Expanded(child: _achievementCard(
                          icon: Icons.visibility_rounded,
                          label: 'Live',
                          sub: 'Searchable now',
                          color: const Color(0xFF3B6D11))),
                        const SizedBox(width: 10),
                        Expanded(child: _achievementCard(
                          icon: Icons.verified_rounded,
                          label: 'Saved',
                          sub: 'In database',
                          color: const Color(0xFF185FA5))),
                      ]),
                      const SizedBox(height: 24),

                      // Actions
                      SizedBox(width: double.infinity, height: 52,
                        child: ElevatedButton(
                          onPressed: () => Navigator.pushReplacement(
  context,
  MaterialPageRoute(
    builder: (_) => OwnerDashboardScreen(
      ownerName: widget.ownerName,
      ownerEmail: '',
      hostelName: widget.hostelName,
      hostelType: '',
      city: '',
      address: '',
    ),
  ),
),
                          style: ElevatedButton.styleFrom(backgroundColor: AppTheme.teal, foregroundColor: Colors.white,
                            elevation: 0, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                          child: Text('Go to Home', style: GoogleFonts.dmSans(fontSize: 16, fontWeight: FontWeight.w600)),
                        )),
                      const SizedBox(height: 12),
                      SizedBox(width: double.infinity, height: 48,
                        child: OutlinedButton(
                          onPressed: () => Navigator.pushNamedAndRemoveUntil(context, '/', (_) => false),
                          style: OutlinedButton.styleFrom(foregroundColor: AppTheme.teal,
                            side: const BorderSide(color: AppTheme.teal, width: 1),
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                          child: Text('Add Another Hostel', style: GoogleFonts.dmSans(fontSize: 15, fontWeight: FontWeight.w600)),
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

  Widget _achievementCard({required IconData icon, required String label, required String sub, required Color color}) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.06), borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.18), width: 0.5)),
      child: Column(children: [
        Icon(icon, size: 22, color: color),
        const SizedBox(height: 6),
        Text(label, style: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w700, color: color)),
        Text(sub, style: GoogleFonts.dmSans(fontSize: 10, color: AppTheme.textMuted), textAlign: TextAlign.center),
      ]),
    );
  }
}