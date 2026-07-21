import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import 'booking_screen.dart';
import 'ai_room_search_screen.dart';

/// Shown when a student taps "Book Now".
/// Lets them choose between AI-assisted booking or browsing manually.
class BookingChoiceScreen extends StatelessWidget {
  final int hostelId;
  final String hostelName;
  final String city;
  final String address;
  final String studentName;
  final String studentPhone;
  final String studentEmail;
  final String studentCnic;
  final String studentUniversity;
  final String parentName;
  final String parentPhone;
  final String parentRelation;

  const BookingChoiceScreen({
    super.key,
    required this.hostelId,
    required this.hostelName,
    required this.city,
    required this.address,
    this.studentName = '',
    this.studentPhone = '',
    this.studentEmail = '',
    this.studentCnic = '',
    this.studentUniversity = '',
    this.parentName = '',
    this.parentPhone = '',
    this.parentRelation = '',
  });

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        body: Column(
          children: [
            _buildHeader(context),
            Expanded(child: _buildBody(context)),
          ],
        ),
      ),
    );
  }

  // ── Header ──────────────────────────────────────────────────────
  Widget _buildHeader(BuildContext context) {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [AppTheme.tealDeep, AppTheme.teal, Color(0xFF1D9E75)],
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 16, 20, 28),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Back button
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Container(
                  width: 38,
                  height: 38,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    shape: BoxShape.circle,
                    border: Border.all(
                        color: Colors.white.withOpacity(0.3), width: 1.5),
                  ),
                  child: const Icon(Icons.arrow_back_rounded,
                      color: Colors.white, size: 18),
                ),
              ),
              const SizedBox(height: 18),
              Text(
                'How would you like\nto book?',
                style: GoogleFonts.playfairDisplay(
                  fontSize: 26,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                  height: 1.2,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                hostelName,
                style: GoogleFonts.dmSans(
                  fontSize: 13,
                  color: Colors.white.withOpacity(0.65),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Body ────────────────────────────────────────────────────────
  Widget _buildBody(BuildContext context) {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 28, 20, 32),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ── AI Option ────────────────────────────────────────
          _ChoiceCard(
            icon: Icons.auto_awesome_rounded,
            iconBg: const Color(0xFF0A3D3F),
            iconColor: AppTheme.tealMint,
            badge: 'RECOMMENDED',
            badgeColor: AppTheme.green,
            title: 'AI-Powered Room Search',
            subtitle:
                'Let our hybrid ML engine find the perfect room based on your lifestyle, budget, and preferences.',
            bullets: const [
              'Content-Based + Collaborative Filtering',
              'Adaptive matching from trained model',
              'Ranked results with match scores',
              'Smart preference profiling',
            ],
            bulletColor: AppTheme.teal,
            borderColor: AppTheme.teal,
            buttonLabel: 'Search with AI',
            buttonColor: AppTheme.teal,
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => AiRoomSearchScreen(
                    hostelId: hostelId,
                    hostelName: hostelName,
                    city: city,
                    address: address,
                    studentName: studentName,
                    studentPhone: studentPhone,
                    studentEmail: studentEmail,
                    studentCnic: studentCnic,
                    studentUniversity: studentUniversity,
                  ),
                ),
              );
            },
          ),
          const SizedBox(height: 18),

          // Divider with OR label
          Row(children: [
            Expanded(
                child: Divider(color: AppTheme.border, thickness: 0.5)),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 14),
              child: Text('OR',
                  style: GoogleFonts.dmSans(
                      fontSize: 12,
                      color: AppTheme.textMuted,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 1.2)),
            ),
            Expanded(
                child: Divider(color: AppTheme.border, thickness: 0.5)),
          ]),
          const SizedBox(height: 18),

          // ── Manual Option ─────────────────────────────────────
          _ChoiceCard(
            icon: Icons.edit_note_rounded,
            iconBg: const Color(0xFF1A2B5E),
            iconColor: const Color(0xFF8EAFF5),
            badge: 'MANUAL',
            badgeColor: const Color(0xFF185FA5),
            title: 'Book Yourself',
            subtitle:
                'Browse and select your room type, set dates, and fill in your own details manually.',
            bullets: const [
              'Choose room type & duration',
              'Pick your own check-in date',
              'Enter personal details',
              'Direct booking request',
            ],
            bulletColor: const Color(0xFF185FA5),
            borderColor: const Color(0xFF185FA5),
            buttonLabel: 'Book Manually',
            buttonColor: const Color(0xFF185FA5),
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => BookingScreen(
                    hostelId: hostelId,
                    hostelName: hostelName,
                    city: city,
                    address: address,
                    studentName: studentName,
                    studentPhone: studentPhone,
                    studentEmail: studentEmail,
                    studentCnic: studentCnic,
                    studentUniversity: studentUniversity,
                    parentName: parentName,
                    parentPhone: parentPhone,
                    parentRelation: parentRelation,
                  ),
                ),
              );
            },
          ),
          const SizedBox(height: 28),

          // Info note
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: AppTheme.amber.withOpacity(0.07),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(
                  color: AppTheme.amber.withOpacity(0.25), width: 0.5),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Icon(Icons.info_outline_rounded,
                    size: 16, color: AppTheme.amber),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Both options result in a booking request sent to the hostel owner for approval within 24 hours.',
                    style: GoogleFonts.dmSans(
                        fontSize: 12,
                        color: const Color(0xFF7A5200),
                        height: 1.5),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

// ── Choice Card Widget ──────────────────────────────────────────────────────
class _ChoiceCard extends StatelessWidget {
  final IconData icon;
  final Color iconBg;
  final Color iconColor;
  final String badge;
  final Color badgeColor;
  final String title;
  final String subtitle;
  final List<String> bullets;
  final Color bulletColor;
  final Color borderColor;
  final String buttonLabel;
  final Color buttonColor;
  final VoidCallback onTap;

  const _ChoiceCard({
    required this.icon,
    required this.iconBg,
    required this.iconColor,
    required this.badge,
    required this.badgeColor,
    required this.title,
    required this.subtitle,
    required this.bullets,
    required this.bulletColor,
    required this.borderColor,
    required this.buttonLabel,
    required this.buttonColor,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: borderColor.withOpacity(0.3), width: 1.5),
        boxShadow: [
          BoxShadow(
              color: borderColor.withOpacity(0.08),
              blurRadius: 16,
              offset: const Offset(0, 4)),
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Top row: icon + badge
            Row(
              children: [
                Container(
                  width: 52,
                  height: 52,
                  decoration: BoxDecoration(
                    color: iconBg,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Icon(icon, color: iconColor, size: 26),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: badgeColor.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: Text(badge,
                            style: GoogleFonts.dmSans(
                                fontSize: 10,
                                fontWeight: FontWeight.w700,
                                color: badgeColor,
                                letterSpacing: 0.8)),
                      ),
                      const SizedBox(height: 5),
                      Text(title,
                          style: GoogleFonts.dmSans(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: AppTheme.textPrimary)),
                    ],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 14),

            // Subtitle
            Text(subtitle,
                style: GoogleFonts.dmSans(
                    fontSize: 13,
                    color: AppTheme.textSecondary,
                    height: 1.5)),
            const SizedBox(height: 14),

            // Bullet points
            ...bullets.map((b) => Padding(
                  padding: const EdgeInsets.only(bottom: 7),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Container(
                        width: 6,
                        height: 6,
                        margin: const EdgeInsets.only(top: 5, right: 10),
                        decoration: BoxDecoration(
                            color: bulletColor, shape: BoxShape.circle),
                      ),
                      Expanded(
                        child: Text(b,
                            style: GoogleFonts.dmSans(
                                fontSize: 12,
                                color: AppTheme.textSecondary)),
                      ),
                    ],
                  ),
                )),
            const SizedBox(height: 16),

            // Button
            SizedBox(
              width: double.infinity,
              height: 48,
              child: ElevatedButton(
                onPressed: onTap,
                style: ElevatedButton.styleFrom(
                  backgroundColor: buttonColor,
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14)),
                ),
                child: Text(buttonLabel,
                    style: GoogleFonts.dmSans(
                        fontSize: 14, fontWeight: FontWeight.w600)),
              ),
            ),
          ],
        ),
      ),
    );
  }
}