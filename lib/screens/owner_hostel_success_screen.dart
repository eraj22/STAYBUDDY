import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';

import 'owner/owner_shell_screen.dart';
import 'owner_hostel_info_screen.dart';

// =============================================================================
// SUCCESS SCREEN COLORS
// =============================================================================

const Color _successDarkTeal = Color(0xFF04383B);
const Color _successDeepTeal = Color(0xFF021F22);
const Color _successGreen = Color(0xFF54C49A);

// =============================================================================
// OWNER HOSTEL SUCCESS SCREEN
// =============================================================================

class OwnerHostelSuccessScreen extends StatefulWidget {
  final String ownerName;
  final String ownerEmail;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;

  const OwnerHostelSuccessScreen({
    super.key,
    required this.ownerName,
    required this.ownerEmail,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
  });

  @override
  State<OwnerHostelSuccessScreen> createState() =>
      _OwnerHostelSuccessScreenState();
}

class _OwnerHostelSuccessScreenState extends State<OwnerHostelSuccessScreen>
    with TickerProviderStateMixin {
  late final AnimationController _entranceController;
  late final AnimationController _pulseController;

  late final Animation<double> _iconScale;
  late final Animation<double> _iconOpacity;

  late final Animation<double> _contentOpacity;
  late final Animation<double> _contentSlide;

  late final Animation<double> _pulse;

  @override
  void initState() {
    super.initState();

    // -------------------------------------------------------------------------
    // Entrance animation
    // -------------------------------------------------------------------------

    _entranceController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1100),
    );

    _iconScale = Tween<double>(begin: 0.35, end: 1.0).animate(
      CurvedAnimation(
        parent: _entranceController,
        curve: const Interval(0.0, 0.58, curve: Curves.elasticOut),
      ),
    );

    _iconOpacity = CurvedAnimation(
      parent: _entranceController,
      curve: const Interval(0.0, 0.28, curve: Curves.easeOut),
    );

    _contentOpacity = CurvedAnimation(
      parent: _entranceController,
      curve: const Interval(0.40, 1.0, curve: Curves.easeOut),
    );

    _contentSlide = Tween<double>(begin: 28, end: 0).animate(
      CurvedAnimation(
        parent: _entranceController,
        curve: const Interval(0.40, 1.0, curve: Curves.easeOutCubic),
      ),
    );

    // -------------------------------------------------------------------------
    // Subtle infinite pulse
    // -------------------------------------------------------------------------

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    );

    _pulse = Tween<double>(begin: 0.96, end: 1.04).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

    _entranceController.forward();

    _pulseController.repeat(reverse: true);
  }

  @override
  void dispose() {
    _entranceController.dispose();
    _pulseController.dispose();
    super.dispose();
  }

  // ===========================================================================
  // NAVIGATION
  // ===========================================================================

  void _goToHome() {
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerShellScreen(
          ownerName: widget.ownerName,
          ownerEmail: widget.ownerEmail,
          hostelName: widget.hostelName,
          hostelType: widget.hostelType,
          city: widget.city,
          address: widget.address,
        ),
      ),
      (route) => false,
    );
  }

  void _addAnotherHostel() {
    Navigator.pushAndRemoveUntil(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerHostelInfoScreen(
          ownerName: widget.ownerName,
          ownerEmail: widget.ownerEmail,
        ),
      ),
      (route) => false,
    );
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.30,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Center(
                child: AnimatedBuilder(
                  animation: _entranceController,
                  builder: (context, child) {
                    return Transform.translate(
                      offset: Offset(0, _contentSlide.value),
                      child: Opacity(
                        opacity: _contentOpacity.value,
                        child: child,
                      ),
                    );
                  },
                  child: _buildSuccessCard(),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // MAIN SUCCESS CARD
  // ===========================================================================

  Widget _buildSuccessCard() {
    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(maxWidth: 520),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF06484B).withOpacity(0.91),
            const Color(0xFF023336).withOpacity(0.93),
            _successDeepTeal.withOpacity(0.96),
          ],
        ),
        borderRadius: BorderRadius.circular(30),
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.25)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.35),
            blurRadius: 30,
            offset: const Offset(0, 14),
          ),
        ],
      ),
      // Retain a bounded viewport so the card measures reliably inside the
      // phone shell, but disable user scrolling: this completion page fits on
      // one screen.
      child: SingleChildScrollView(
        primary: false,
        physics: const NeverScrollableScrollPhysics(),
        padding: const EdgeInsets.fromLTRB(22, 26, 22, 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // -----------------------------------------------------------------
            // Small status badge
            // -----------------------------------------------------------------
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: _successGreen.withOpacity(0.12),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: _successGreen.withOpacity(0.28)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Icon(
                    Icons.verified_rounded,
                    color: _successGreen,
                    size: 14,
                  ),
                  const SizedBox(width: 6),
                  Text(
                    'Registration Complete',
                    style: GoogleFonts.dmSans(
                      color: _successGreen,
                      fontSize: 10,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 22),

            // -----------------------------------------------------------------
            // Animated success icon
            // -----------------------------------------------------------------
            _buildSuccessIcon(),

            const SizedBox(height: 22),

            // -----------------------------------------------------------------
            // Heading
            // -----------------------------------------------------------------
            Text(
              'Hostel Listed!',
              textAlign: TextAlign.center,
              style: GoogleFonts.playfairDisplay(
                color: Colors.white,
                fontSize: 31,
                height: 1.05,
                fontWeight: FontWeight.w700,
              ),
            ),

            const SizedBox(height: 8),

            Text(
              'Your hostel registration has been completed successfully.',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                color: Colors.white.withOpacity(0.58),
                fontSize: 11,
                height: 1.45,
              ),
            ),

            const SizedBox(height: 18),

            // -----------------------------------------------------------------
            // Hostel information
            // -----------------------------------------------------------------
            _buildHostelCard(),

            const SizedBox(height: 18),

            // -----------------------------------------------------------------
            // Status / achievements
            // -----------------------------------------------------------------
            _buildStatusSection(),

            const SizedBox(height: 20),

            // -----------------------------------------------------------------
            // Primary button
            // -----------------------------------------------------------------
            SizedBox(
              width: double.infinity,
              height: 51,
              child: ElevatedButton(
                onPressed: _goToHome,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.teal,
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.dashboard_rounded, size: 18),
                    const SizedBox(width: 9),
                    Text(
                      'Go to Owner Dashboard',
                      style: GoogleFonts.dmSans(
                        fontSize: 14,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(width: 8),
                    const Icon(Icons.arrow_forward_rounded, size: 17),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 10),

            // -----------------------------------------------------------------
            // Add another hostel
            // -----------------------------------------------------------------
            SizedBox(
              width: double.infinity,
              height: 48,
              child: OutlinedButton(
                onPressed: _addAnotherHostel,
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppTheme.tealMint,
                  side: BorderSide(color: AppTheme.tealMint.withOpacity(0.32)),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.add_business_rounded, size: 17),
                    const SizedBox(width: 8),
                    Text(
                      'Add Another Hostel',
                      style: GoogleFonts.dmSans(
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ),

            const SizedBox(height: 16),

            // -----------------------------------------------------------------
            // Bottom message
            // -----------------------------------------------------------------
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.shield_outlined,
                  color: Colors.white.withOpacity(0.34),
                  size: 13,
                ),
                const SizedBox(width: 6),
                Flexible(
                  child: Text(
                    'Your hostel information is safely stored with StayBuddy.',
                    textAlign: TextAlign.center,
                    style: GoogleFonts.dmSans(
                      color: Colors.white.withOpacity(0.34),
                      fontSize: 8.5,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // SUCCESS ICON
  // ===========================================================================

  Widget _buildSuccessIcon() {
    return AnimatedBuilder(
      animation: Listenable.merge([_entranceController, _pulseController]),
      builder: (context, child) {
        return FadeTransition(
          opacity: _iconOpacity,
          child: ScaleTransition(
            scale: _iconScale,
            child: Transform.scale(scale: _pulse.value, child: child),
          ),
        );
      },
      child: SizedBox(
        width: 128,
        height: 128,
        child: Stack(
          alignment: Alignment.center,
          children: [
            Container(
              width: 128,
              height: 128,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: AppTheme.tealMint.withOpacity(0.04),
                border: Border.all(color: AppTheme.tealMint.withOpacity(0.08)),
              ),
            ),

            Container(
              width: 100,
              height: 100,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: AppTheme.teal.withOpacity(0.13),
                border: Border.all(color: AppTheme.tealMint.withOpacity(0.14)),
              ),
            ),

            Container(
              width: 74,
              height: 74,
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                  colors: [AppTheme.teal, _successDarkTeal],
                ),
                shape: BoxShape.circle,
                border: Border.all(color: AppTheme.tealMint.withOpacity(0.35)),
                boxShadow: [
                  BoxShadow(
                    color: AppTheme.teal.withOpacity(0.40),
                    blurRadius: 24,
                    spreadRadius: 2,
                  ),
                ],
              ),
              child: const Icon(
                Icons.check_rounded,
                color: Colors.white,
                size: 40,
              ),
            ),

            Positioned(
              right: 15,
              top: 18,
              child: Container(
                width: 27,
                height: 27,
                decoration: BoxDecoration(
                  color: _successGreen,
                  shape: BoxShape.circle,
                  border: Border.all(color: _successDeepTeal, width: 3),
                ),
                child: const Icon(
                  Icons.location_on_rounded,
                  color: Colors.white,
                  size: 14,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // HOSTEL CARD
  // ===========================================================================

  Widget _buildHostelCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.teal.withOpacity(0.16),
        borderRadius: BorderRadius.circular(17),
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.20)),
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.30),
              borderRadius: BorderRadius.circular(14),
            ),
            child: const Icon(
              Icons.apartment_rounded,
              color: AppTheme.tealMint,
              size: 24,
            ),
          ),

          const SizedBox(width: 12),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.hostelName,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 13.5,
                    fontWeight: FontWeight.w800,
                  ),
                ),

                const SizedBox(height: 4),

                Text(
                  '${_formatHostelType(widget.hostelType)} · ${widget.city}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: AppTheme.tealMint,
                    fontSize: 9.5,
                    fontWeight: FontWeight.w600,
                  ),
                ),

                if (widget.address.trim().isNotEmpty) ...[
                  const SizedBox(height: 3),
                  Text(
                    widget.address,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.dmSans(
                      color: Colors.white.withOpacity(0.44),
                      fontSize: 8.5,
                    ),
                  ),
                ],
              ],
            ),
          ),

          const SizedBox(width: 8),

          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
            decoration: BoxDecoration(
              color: _successGreen.withOpacity(0.14),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text(
              'LIVE',
              style: GoogleFonts.dmSans(
                color: _successGreen,
                fontSize: 8,
                letterSpacing: 0.6,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // STATUS SECTION
  // ===========================================================================

  Widget _buildStatusSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Icon(
              Icons.auto_awesome_rounded,
              color: AppTheme.tealMint,
              size: 15,
            ),
            const SizedBox(width: 7),
            Text(
              'Your hostel is ready',
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 11.5,
                fontWeight: FontWeight.w800,
              ),
            ),
          ],
        ),

        const SizedBox(height: 10),

        Row(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: _achievementCard(
                icon: Icons.map_rounded,
                label: 'On Map',
                sub: 'Students can find it',
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _achievementCard(
                icon: Icons.visibility_rounded,
                label: 'Visible',
                sub: 'Profile is discoverable',
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _achievementCard(
                icon: Icons.verified_user_rounded,
                label: 'Registered',
                sub: 'Setup is complete',
              ),
            ),
          ],
        ),
      ],
    );
  }

  // ===========================================================================
  // ACHIEVEMENT CARD
  // ===========================================================================

  Widget _achievementCard({
    required IconData icon,
    required String label,
    required String sub,
  }) {
    return Container(
      constraints: const BoxConstraints(minHeight: 108),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 12),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.18),
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: Colors.white.withOpacity(0.10)),
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.22),
              borderRadius: BorderRadius.circular(11),
            ),
            child: Icon(icon, color: AppTheme.tealMint, size: 18),
          ),

          const SizedBox(height: 8),

          Text(
            label,
            textAlign: TextAlign.center,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 10.5,
              fontWeight: FontWeight.w800,
            ),
          ),

          const SizedBox(height: 3),

          Text(
            sub,
            textAlign: TextAlign.center,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.43),
              fontSize: 7.8,
              height: 1.3,
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // HOSTEL TYPE
  // ===========================================================================

  String _formatHostelType(String type) {
    final normalized = type.trim().toLowerCase();

    if (normalized == 'girls' ||
        normalized == 'girl' ||
        normalized == 'female') {
      return 'Girls Hostel';
    }

    if (normalized == 'boys' || normalized == 'boy' || normalized == 'male') {
      return 'Boys Hostel';
    }

    if (normalized == 'co-ed' ||
        normalized == 'coed' ||
        normalized == 'mixed') {
      return 'Co-ed Hostel';
    }

    return type;
  }
}
