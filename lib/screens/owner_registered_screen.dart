import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_hostel_info_screen.dart';

class OwnerRegisteredScreen extends StatefulWidget {
  final String name;
  final String email;

  const OwnerRegisteredScreen({
    super.key,
    required this.name,
    required this.email,
  });

  @override
  State<OwnerRegisteredScreen> createState() =>
      _OwnerRegisteredScreenState();
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

  // ===========================================================================
  // INIT
  // ===========================================================================

  @override
  void initState() {
    super.initState();

    _checkAnim = AnimationController(
      vsync: this,
      duration: const Duration(
        milliseconds: 850,
      ),
    );

    _textAnim = AnimationController(
      vsync: this,
      duration: const Duration(
        milliseconds: 650,
      ),
    );

    _checkScale = Tween<double>(
      begin: 0,
      end: 1,
    ).animate(
      CurvedAnimation(
        parent: _checkAnim,
        curve: const Interval(
          0.18,
          0.85,
          curve: Curves.elasticOut,
        ),
      ),
    );

    _checkOpacity = CurvedAnimation(
      parent: _checkAnim,
      curve: const Interval(
        0,
        0.35,
        curve: Curves.easeOut,
      ),
    );

    _ringExpand = Tween<double>(
      begin: 0.68,
      end: 1,
    ).animate(
      CurvedAnimation(
        parent: _checkAnim,
        curve: const Interval(
          0,
          0.65,
          curve: Curves.easeOutCubic,
        ),
      ),
    );

    _textFade = CurvedAnimation(
      parent: _textAnim,
      curve: Curves.easeOut,
    );

    _textSlide = Tween<double>(
      begin: 24,
      end: 0,
    ).animate(
      CurvedAnimation(
        parent: _textAnim,
        curve: Curves.easeOutCubic,
      ),
    );

    _checkAnim.forward().then(
      (_) {
        if (mounted) {
          _textAnim.forward();
        }
      },
    );
  }

  // ===========================================================================
  // DISPOSE
  // ===========================================================================

  @override
  void dispose() {
    _checkAnim.dispose();
    _textAnim.dispose();

    super.dispose();
  }

  // ===========================================================================
  // CONTINUE
  // ===========================================================================

  void _continueToHostel() {
    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerHostelInfoScreen(
          ownerEmail: widget.email,
          ownerName: widget.name,
        ),
      ),
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

          // Background video clear but readable.
          overlayOpacity: 0.24,

          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,

          child: SafeArea(
            child: LayoutBuilder(
              builder: (
                context,
                constraints,
              ) {
                return SingleChildScrollView(
                  physics: const BouncingScrollPhysics(),
                  padding: const EdgeInsets.fromLTRB(
                    18,
                    16,
                    18,
                    24,
                  ),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(
                      minHeight: constraints.maxHeight - 40,
                    ),
                    child: Column(
                      children: [
                        // Gives balanced spacing on taller screens.
                        SizedBox(
                          height: constraints.maxHeight > 760
                              ? 55
                              : 18,
                        ),

                        // =====================================================
                        // SUCCESS HEADING ABOVE CARD
                        // =====================================================

                        _buildTopHeading(),

                        const SizedBox(height: 24),

                        // =====================================================
                        // DARK TEAL GLASS CARD
                        // =====================================================

                        _buildSuccessCard(),

                        const SizedBox(height: 24),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // TOP HEADING
  // ===========================================================================

  Widget _buildTopHeading() {
    return AnimatedBuilder(
      animation: _textAnim,
      builder: (
        context,
        child,
      ) {
        return Transform.translate(
          offset: Offset(
            0,
            _textSlide.value,
          ),
          child: Opacity(
            opacity: _textFade.value,
            child: child,
          ),
        );
      },
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(
              horizontal: 13,
              vertical: 7,
            ),
            decoration: BoxDecoration(
              color: const Color(
                0xFF02383B,
              ).withOpacity(0.55),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(
                color: AppTheme.tealMint.withOpacity(
                  0.25,
                ),
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Icon(
                  Icons.verified_rounded,
                  size: 15,
                  color: AppTheme.tealMint,
                ),
                const SizedBox(width: 7),
                Text(
                  'Owner Registration',
                  style: GoogleFonts.dmSans(
                    color: Colors.white.withOpacity(
                      0.90,
                    ),
                    fontSize: 11.5,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 16),

          Text(
            'You\'re All Set!',
            textAlign: TextAlign.center,
            style: GoogleFonts.playfairDisplay(
              fontSize: 36,
              height: 1.05,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),

          const SizedBox(height: 10),

          Text(
            'Your owner account has been created successfully.',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              fontSize: 12.5,
              height: 1.45,
              fontWeight: FontWeight.w500,
              color: Colors.white.withOpacity(
                0.70,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // SUCCESS CARD
  // ===========================================================================

  Widget _buildSuccessCard() {
    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(
        maxWidth: 420,
      ),
      padding: const EdgeInsets.fromLTRB(
        21,
        18,
        21,
        21,
      ),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(
              0xFF063F42,
            ).withOpacity(0.88),
            const Color(
              0xFF023336,
            ).withOpacity(0.83),
            const Color(
              0xFF021F22,
            ).withOpacity(0.87),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(
          color: AppTheme.tealMint.withOpacity(
            0.24,
          ),
          width: 0.9,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(
              0.30,
            ),
            blurRadius: 24,
            offset: const Offset(
              0,
              10,
            ),
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // ===============================================================
          // HANDLE
          // ===============================================================

          Container(
            width: 38,
            height: 4,
            margin: const EdgeInsets.only(
              bottom: 18,
            ),
            decoration: BoxDecoration(
              color: AppTheme.tealMint.withOpacity(
                0.32,
              ),
              borderRadius: BorderRadius.circular(
                3,
              ),
            ),
          ),

          // ===============================================================
          // SUCCESS ANIMATION
          // ===============================================================

          _buildSuccessAnimation(),

          const SizedBox(height: 18),

          // ===============================================================
          // TEXT
          // ===============================================================

          AnimatedBuilder(
            animation: _textAnim,
            builder: (
              context,
              child,
            ) {
              return Transform.translate(
                offset: Offset(
                  0,
                  _textSlide.value,
                ),
                child: Opacity(
                  opacity: _textFade.value,
                  child: child,
                ),
              );
            },
            child: _buildContent(),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // SUCCESS ANIMATION
  // ===========================================================================

  Widget _buildSuccessAnimation() {
    return AnimatedBuilder(
      animation: _checkAnim,
      builder: (
        context,
        child,
      ) {
        return ScaleTransition(
          scale: _ringExpand,
          child: FadeTransition(
            opacity: _checkOpacity,
            child: Stack(
              alignment: Alignment.center,
              children: [
                // Outer teal glow
                Container(
                  width: 104,
                  height: 104,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: AppTheme.tealMint.withOpacity(
                      0.07,
                    ),
                    border: Border.all(
                      color: AppTheme.tealMint.withOpacity(
                        0.08,
                      ),
                    ),
                  ),
                ),

                // Middle ring
                Container(
                  width: 84,
                  height: 84,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: AppTheme.tealMint.withOpacity(
                      0.10,
                    ),
                    border: Border.all(
                      color: AppTheme.tealMint.withOpacity(
                        0.14,
                      ),
                    ),
                  ),
                ),

                // Check circle
                ScaleTransition(
                  scale: _checkScale,
                  child: Container(
                    width: 64,
                    height: 64,
                    decoration: BoxDecoration(
                      color: AppTheme.teal,
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: AppTheme.tealMint.withOpacity(
                          0.40,
                        ),
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: AppTheme.tealMint.withOpacity(
                            0.22,
                          ),
                          blurRadius: 20,
                          spreadRadius: 1,
                        ),
                      ],
                    ),
                    child: const Icon(
                      Icons.check_rounded,
                      color: Colors.white,
                      size: 35,
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  // ===========================================================================
  // CONTENT
  // ===========================================================================

  Widget _buildContent() {
    return Column(
      children: [
        Text(
          'Profile Created!',
          textAlign: TextAlign.center,
          style: GoogleFonts.playfairDisplay(
            fontSize: 25,
            height: 1.1,
            fontWeight: FontWeight.w700,
            color: Colors.white,
          ),
        ),

        const SizedBox(height: 7),

        // ===============================================================
        // OWNER NAME
        // ===============================================================

        Container(
          padding: const EdgeInsets.symmetric(
            horizontal: 12,
            vertical: 6,
          ),
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(
              0.20,
            ),
            borderRadius: BorderRadius.circular(
              20,
            ),
            border: Border.all(
              color: AppTheme.tealMint.withOpacity(
                0.18,
              ),
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(
                Icons.person_outline_rounded,
                size: 14,
                color: AppTheme.tealMint,
              ),
              const SizedBox(width: 6),
              Flexible(
                child: Text(
                  'Welcome, ${widget.name}!',
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 11.5,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.tealMint,
                  ),
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 12),

        Text(
          'Your owner account is ready.\nNow let\'s set up your hostel.',
          textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(
            fontSize: 11.5,
            height: 1.5,
            color: Colors.white.withOpacity(
              0.60,
            ),
          ),
        ),

        const SizedBox(height: 20),

        // ===============================================================
        // WHAT'S NEXT
        // ===============================================================

        _buildNextCard(),

        const SizedBox(height: 20),

        // ===============================================================
        // BUTTON
        // ===============================================================

        _buildContinueButton(),

        const SizedBox(height: 13),

        // ===============================================================
        // NOTE
        // ===============================================================

        _buildSecurityNote(),
      ],
    );
  }

  // ===========================================================================
  // WHAT'S NEXT CARD
  // ===========================================================================

  Widget _buildNextCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(
        14,
        13,
        14,
        14,
      ),
      decoration: BoxDecoration(
        color: const Color(
          0xFF01272A,
        ).withOpacity(0.58),
        borderRadius: BorderRadius.circular(
          16,
        ),
        border: Border.all(
          color: AppTheme.tealMint.withOpacity(
            0.18,
          ),
        ),
      ),
      child: Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 30,
                height: 30,
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(
                    0.25,
                  ),
                  borderRadius:
                      BorderRadius.circular(
                    9,
                  ),
                ),
                child: const Icon(
                  Icons.route_outlined,
                  size: 16,
                  color: AppTheme.tealMint,
                ),
              ),

              const SizedBox(width: 9),

              Expanded(
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Text(
                      'What\'s next?',
                      style: GoogleFonts.dmSans(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      'Complete your hostel setup',
                      style: GoogleFonts.dmSans(
                        fontSize: 9.5,
                        color: Colors.white.withOpacity(
                          0.45,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 13),

          _nextItem(
            number: '1',
            icon: Icons.business_rounded,
            label: 'Add hostel information',
          ),

          const SizedBox(height: 9),

          _nextItem(
            number: '2',
            icon: Icons.location_on_outlined,
            label: 'Pin hostel location',
          ),

          const SizedBox(height: 9),

          _nextItem(
            number: '3',
            icon: Icons.bed_outlined,
            label: 'Set room & bed details',
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // NEXT ITEM
  // ===========================================================================

  Widget _nextItem({
    required String number,
    required IconData icon,
    required String label,
  }) {
    return Row(
      children: [
        Container(
          width: 30,
          height: 30,
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(
              0.18,
            ),
            shape: BoxShape.circle,
            border: Border.all(
              color: AppTheme.tealMint.withOpacity(
                0.15,
              ),
            ),
          ),
          child: Icon(
            icon,
            size: 14,
            color: AppTheme.tealMint,
          ),
        ),

        const SizedBox(width: 10),

        Expanded(
          child: Text(
            label,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: GoogleFonts.dmSans(
              fontSize: 11,
              height: 1.3,
              fontWeight: FontWeight.w500,
              color: Colors.white.withOpacity(
                0.76,
              ),
            ),
          ),
        ),

        const SizedBox(width: 8),

        Container(
          width: 22,
          height: 22,
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: AppTheme.tealMint.withOpacity(
              0.10,
            ),
            shape: BoxShape.circle,
          ),
          child: Text(
            number,
            style: GoogleFonts.dmSans(
              fontSize: 9,
              fontWeight: FontWeight.w800,
              color: AppTheme.tealMint,
            ),
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // CONTINUE BUTTON
  // ===========================================================================

  Widget _buildContinueButton() {
    return SizedBox(
      width: double.infinity,
      height: 50,
      child: ElevatedButton(
        onPressed: _continueToHostel,
        style: ElevatedButton.styleFrom(
          backgroundColor: AppTheme.teal,
          foregroundColor: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(
              15,
            ),
          ),
        ),
        child: Row(
          mainAxisAlignment:
              MainAxisAlignment.center,
          children: [
            const Icon(
              Icons.apartment_rounded,
              size: 18,
            ),

            const SizedBox(width: 8),

            Text(
              'Add My Hostel',
              style: GoogleFonts.dmSans(
                fontSize: 14.5,
                fontWeight: FontWeight.w800,
              ),
            ),

            const SizedBox(width: 8),

            Container(
              width: 25,
              height: 25,
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(
                  0.15,
                ),
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.arrow_forward_rounded,
                size: 15,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // SECURITY / SAVED NOTE
  // ===========================================================================

  Widget _buildSecurityNote() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(
        horizontal: 11,
        vertical: 9,
      ),
      decoration: BoxDecoration(
        color: const Color(
          0xFF01272A,
        ).withOpacity(0.48),
        borderRadius: BorderRadius.circular(
          13,
        ),
        border: Border.all(
          color: AppTheme.tealMint.withOpacity(
            0.15,
          ),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 29,
            height: 29,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(
                0.20,
              ),
              borderRadius: BorderRadius.circular(
                9,
              ),
            ),
            child: const Icon(
              Icons.verified_user_outlined,
              size: 15,
              color: AppTheme.tealMint,
            ),
          ),

          const SizedBox(width: 9),

          Expanded(
            child: Text(
              'Your owner profile has been verified and saved successfully.',
              style: GoogleFonts.dmSans(
                fontSize: 9.5,
                height: 1.35,
                color: Colors.white.withOpacity(
                  0.55,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}