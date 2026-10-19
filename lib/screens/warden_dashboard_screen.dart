import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';

import 'warden_login_screen.dart';
import 'warden_qr_attendance_screen.dart';
import 'chatbot_screen.dart';
import 'warden_use_cases.dart';

class WardenDashboardScreen extends StatefulWidget {
  final String wardenName;
  final String hostelName;
  final String hostelId;

  const WardenDashboardScreen({
    super.key,
    required this.wardenName,
    required this.hostelName,
    required this.hostelId,
  });

  @override
  State<WardenDashboardScreen> createState() => _WardenDashboardScreenState();
}

class _WardenDashboardScreenState extends State<WardenDashboardScreen>
    with SingleTickerProviderStateMixin {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  bool _loading = true;

  Map<String, dynamic> _stats = {
    'students': 62,
    'complaints': 4,
    'attendance': 0,
    'available': 18,
  };

  late final AnimationController _entryController;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  final List<Map<String, dynamic>> _activity = [
    {
      'icon': Icons.report_problem_rounded,
      'title': 'New complaint received',
      'subtitle': 'WiFi issue · Room 102 · 2 hrs ago',
    },
    {
      'icon': Icons.how_to_reg_rounded,
      'title': 'Attendance updated',
      'subtitle': 'Evening attendance completed',
    },
    {
      'icon': Icons.bed_rounded,
      'title': 'Bed became available',
      'subtitle': 'Room 301 · Bed 02',
    },
    {
      'icon': Icons.campaign_rounded,
      'title': 'Announcement posted',
      'subtitle': 'Mess timing update',
    },
  ];

  // ===========================================================================
  // LIFECYCLE
  // ===========================================================================

  @override
  void initState() {
    super.initState();

    _entryController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );

    _fade = CurvedAnimation(parent: _entryController, curve: Curves.easeOut);

    _slide = Tween<double>(begin: 24, end: 0).animate(
      CurvedAnimation(parent: _entryController, curve: Curves.easeOutCubic),
    );

    _entryController.forward();

    _loadDashboard();
  }

  @override
  void dispose() {
    _entryController.dispose();

    super.dispose();
  }

  // ===========================================================================
  // DASHBOARD DATA
  // ===========================================================================

  Future<void> _loadDashboard() async {
    try {
      final data = await Api().getWardenDashboard();
      var attendancePercentage = data['attendance'] ?? 0;
      try {
        final assigned = await Api().getWardenAssignedHostel();
        final hostel = Map<String, dynamic>.from(assigned['hostel'] as Map);
        final today = DateTime.now();
        final date =
            '${today.year.toString().padLeft(4, '0')}-${today.month.toString().padLeft(2, '0')}-${today.day.toString().padLeft(2, '0')}';
        final gateAttendance = await Api().getQrAttendanceDashboard(
          int.parse(hostel['id'].toString()),
          date: date,
        );
        final residents = List<dynamic>.from(
          gateAttendance['residents'] as List? ?? const [],
        );
        final residentMaps = residents
            .whereType<Map>()
            .map((resident) => Map<String, dynamic>.from(resident))
            .toList();
        final residentIds = residentMaps
            .map((resident) => resident['student_id']?.toString())
            .whereType<String>()
            .toSet();
        final recordedIds = List<dynamic>.from(
          gateAttendance['recent_events'] as List? ?? const [],
        )
            .map((event) { final value = Map<String, dynamic>.from(event as Map); return (value['student_user_id'] ?? value['student_id'] ?? value['user_id'])?.toString(); })
            .whereType<String>()
            .where(residentIds.contains)
            .toSet();
        final recordedNames = List<dynamic>.from(
          gateAttendance['recent_events'] as List? ?? const [],
        )
            .map((event) => Map<String, dynamic>.from(event as Map)['student_name']?.toString().trim().toLowerCase())
            .whereType<String>()
            .where((name) => name.isNotEmpty)
            .toSet();
        final scannedStudents = residentMaps.where((resident) {
          final id = resident['student_id']?.toString();
          final name = resident['name']?.toString().trim().toLowerCase() ?? '';
          return recordedIds.contains(id) || recordedNames.contains(name);
        }).length;
        // A student with both an IN and an OUT is counted once for the day.
        attendancePercentage = residentIds.isEmpty
            ? 0
            : ((scannedStudents / residentIds.length) * 100).round();
      } catch (error) {
        debugPrint('Unified attendance percentage fetch failed: $error');
      }

      if (!mounted) return;

      setState(() {
        _stats = {
          'students': data['students'] ?? 62,
          'complaints': data['complaints'] ?? 4,
          'attendance': attendancePercentage,
          'available': data['available_beds'] ?? 18,
        };

        _loading = false;
      });
    } catch (error) {
      debugPrint('Warden dashboard fetch failed: $error');

      if (!mounted) return;

      setState(() {
        _loading = false;
      });
    }
  }

  // ===========================================================================
  // HELPERS
  // ===========================================================================

  String get _firstName {
    final value = widget.wardenName.trim();

    if (value.isEmpty) {
      return 'Warden';
    }

    return value.split(' ').first;
  }

  String get _initial {
    final value = widget.wardenName.trim();

    if (value.isEmpty) {
      return 'W';
    }

    return value[0].toUpperCase();
  }

  String get _greeting {
    final hour = DateTime.now().hour;

    if (hour < 12) {
      return 'Good morning';
    }

    if (hour < 17) {
      return 'Good afternoon';
    }

    return 'Good evening';
  }

  Future<void> _pushScreen(Widget screen) async {
    await Navigator.push(context, MaterialPageRoute(builder: (_) => screen));

    if (mounted) {
      await _loadDashboard();
    }
  }

  void _openDrawerScreen(Widget screen) {
    Navigator.pop(context);

    Future.delayed(const Duration(milliseconds: 120), () {
      if (!mounted) return;

      _pushScreen(screen);
    });
  }

  // ===========================================================================
  // LOGOUT
  // ===========================================================================

  Future<void> _logout() async {
    await AuthStore.clear();

    if (!mounted) return;

    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => const WardenLoginScreen()),
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
        key: _scaffoldKey,
        backgroundColor: Colors.transparent,

        drawer: _buildDrawer(),

        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.46,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,

          child: SafeArea(
            child: AnimatedBuilder(
              animation: _entryController,

              builder: (context, child) {
                return Transform.translate(
                  offset: Offset(0, _slide.value),
                  child: Opacity(opacity: _fade.value, child: child),
                );
              },

              child: RefreshIndicator(
                color: AppTheme.teal,

                onRefresh: _loadDashboard,

                child: CustomScrollView(
                  physics: const AlwaysScrollableScrollPhysics(),

                  slivers: [
                    SliverToBoxAdapter(child: _buildHeader()),

                    SliverPadding(
                      padding: const EdgeInsets.fromLTRB(16, 20, 16, 0),
                      sliver: SliverToBoxAdapter(child: _buildStatsSection()),
                    ),

                    SliverPadding(
                      padding: const EdgeInsets.fromLTRB(16, 24, 16, 0),
                      sliver: SliverToBoxAdapter(child: _buildQuickActions()),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // HEADER
  // ===========================================================================

  Widget _buildHeader() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ============================================================
          // TOP ROW
          // Bell / notification icon intentionally removed
          // ============================================================
          SizedBox(
            height: 46,
            width: double.infinity,
            child: Stack(
              alignment: Alignment.center,
              children: [
                Align(
                  alignment: Alignment.centerLeft,
                  child: Container(
                    width: 46,
                    height: 46,
                    clipBehavior: Clip.antiAlias,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(
                        color: AppTheme.tealMint.withOpacity(0.75),
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withOpacity(0.18),
                          blurRadius: 10,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Transform.scale(
                      // Keep a little breathing room around the mark inside its badge.
                      scale: 0.82,
                      child: Image.asset(
                        'assets/images/logo.png',
                        fit: BoxFit.contain,
                        semanticLabel: 'StayBuddy logo',
                      ),
                    ),
                  ),
                ),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 12,
                    vertical: 8,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.28),
                    borderRadius: BorderRadius.circular(22),
                    border: Border.all(color: Colors.white.withOpacity(0.18)),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(
                        Icons.verified_user_rounded,
                        size: 14,
                        color: AppTheme.tealMint,
                      ),
                      const SizedBox(width: 6),
                      Text(
                        'Warden Portal',
                        style: GoogleFonts.dmSans(
                          color: Colors.white,
                          fontSize: 10.5,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
                Align(
                  alignment: Alignment.centerRight,
                  child: _glassRoundButton(
                    icon: Icons.menu_rounded,
                    onTap: () => _scaffoldKey.currentState?.openDrawer(),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 24),

          // ============================================================
          // WARDEN PROFILE
          // ============================================================
          Row(
            children: [
              Container(
                width: 60,
                height: 60,
                decoration: BoxDecoration(
                  color: AppTheme.teal,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: AppTheme.teal.withOpacity(0.32),
                      blurRadius: 18,
                      offset: const Offset(0, 7),
                    ),
                  ],
                ),
                child: Center(
                  child: Text(
                    _initial,
                    style: GoogleFonts.playfairDisplay(
                      color: Colors.white,
                      fontSize: 28,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ),
              ),

              const SizedBox(width: 14),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _greeting,
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.70),
                        fontSize: 10.5,
                      ),
                    ),

                    const SizedBox(height: 3),

                    Text(
                      _firstName,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.playfairDisplay(
                        color: Colors.white,
                        fontSize: 30,
                        height: 1,
                        fontWeight: FontWeight.w700,
                      ),
                    ),

                    const SizedBox(height: 5),

                    Text(
                      'Here\'s what\'s happening at your hostel.',
                      maxLines: 2,
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.68),
                        fontSize: 10,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 22),

          // ============================================================
          // HOSTEL CARD
          // ============================================================
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(15),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.32),
              borderRadius: BorderRadius.circular(22),
              border: Border.all(color: Colors.white.withOpacity(0.18)),
            ),
            child: Row(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: AppTheme.teal.withOpacity(0.20),
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: const Icon(
                    Icons.apartment_rounded,
                    color: AppTheme.tealMint,
                    size: 23,
                  ),
                ),

                const SizedBox(width: 13),

                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Managing hostel',
                        style: GoogleFonts.dmSans(
                          color: AppTheme.tealMint,
                          fontSize: 9,
                          fontWeight: FontWeight.w600,
                        ),
                      ),

                      const SizedBox(height: 3),

                      Text(
                        widget.hostelName,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          color: Colors.white,
                          fontSize: 13.5,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(width: 8),

                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 6,
                  ),
                  decoration: BoxDecoration(
                    color: AppTheme.teal.withOpacity(0.18),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: AppTheme.tealMint.withOpacity(0.30),
                    ),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 6,
                        height: 6,
                        decoration: const BoxDecoration(
                          color: AppTheme.tealMint,
                          shape: BoxShape.circle,
                        ),
                      ),

                      const SizedBox(width: 6),

                      Text(
                        'Active',
                        style: GoogleFonts.dmSans(
                          color: AppTheme.tealMint,
                          fontSize: 9.5,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _glassRoundButton({
    required IconData icon,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(50),
      child: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.26),
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white.withOpacity(0.16)),
        ),
        child: Icon(icon, color: Colors.white, size: 21),
      ),
    );
  }

  // ===========================================================================
  // STATS
  // ===========================================================================

  Widget _buildStatsSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionHeader(
          title: 'Today at your hostel',
          subtitle: 'Overview of today\'s activities',
          light: true,
        ),

        const SizedBox(height: 14),

        Row(
          children: [
            Expanded(
              child: _statCard(
                icon: Icons.groups_rounded,
                value: '${_stats['students']}',
                label: 'Students',
              ),
            ),

            const SizedBox(width: 10),

            Expanded(
              child: _statCard(
                icon: Icons.how_to_reg_rounded,
                value: '${_stats['attendance']}%',
                label: 'Attendance',
              ),
            ),
          ],
        ),

        const SizedBox(height: 10),

        Row(
          children: [
            Expanded(
              child: _statCard(
                icon: Icons.warning_amber_rounded,
                value: '${_stats['complaints']}',
                label: 'Open Issues',
              ),
            ),

            const SizedBox(width: 10),

            Expanded(
              child: _statCard(
                icon: Icons.bed_rounded,
                value: '${_stats['available']}',
                label: 'Free Beds',
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _statCard({
    required IconData icon,
    required String value,
    required String label,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.92),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.55)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.10),
            blurRadius: 16,
            offset: const Offset(0, 7),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.10),
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(icon, color: AppTheme.teal, size: 20),
          ),

          const SizedBox(width: 11),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _loading
                    ? Container(
                        width: 34,
                        height: 16,
                        decoration: BoxDecoration(
                          color: AppTheme.teal.withOpacity(0.08),
                          borderRadius: BorderRadius.circular(5),
                        ),
                      )
                    : SizedBox(
                        width: double.infinity,
                        child: FittedBox(
                          fit: BoxFit.scaleDown,
                          alignment: Alignment.centerLeft,
                          child: Text(
                            value,
                            maxLines: 1,
                            softWrap: false,
                            style: GoogleFonts.dmSans(
                              color: const Color(0xFF183636),
                              fontSize: 19,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ),
                      ),

                const SizedBox(height: 2),

                SizedBox(
                  width: double.infinity,
                  child: FittedBox(
                    fit: BoxFit.scaleDown,
                    alignment: Alignment.centerLeft,
                    child: Text(
                      label,
                      style: GoogleFonts.dmSans(
                        color: const Color(0xFF81908F),
                        fontSize: 9.7,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // QUICK ACTIONS
  // ===========================================================================

  Widget _buildQuickActions() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionHeader(
          title: 'Quick actions',
          subtitle: 'Shortcuts to common tasks',
          light: true,
        ),

        const SizedBox(height: 14),

        Row(
          children: [
            Expanded(
              child: _quickAction(
                icon: Icons.campaign_rounded,
                title: 'Announce',
                onTap: () {
                  _pushScreen(
                    WardenAnnouncementsScreen(hostelName: widget.hostelName),
                  );
                },
              ),
            ),

            const SizedBox(width: 9),

            Expanded(
              child: _quickAction(
                icon: Icons.report_problem_outlined,
                title: 'Complaints',
                onTap: () {
                  // NORMAL COMPLAINTS ONLY
                  _pushScreen(
                    WardenComplaintsScreen(hostelName: widget.hostelName),
                  );
                },
              ),
            ),

            const SizedBox(width: 9),

            Expanded(
              child: _quickAction(
                icon: Icons.fact_check_rounded,
                title: 'Attendance',
                onTap: () {
                  _pushScreen(
                    WardenAttendanceScreen(hostelName: widget.hostelName),
                  );
                },
              ),
            ),

            const SizedBox(width: 9),

            Expanded(
              child: _quickAction(
                icon: Icons.payments_outlined,
                title: 'Fees',
                onTap: () {
                  _pushScreen(WardenFeesScreen(hostelName: widget.hostelName));
                },
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _quickAction({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(18),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 6),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.92),
          borderRadius: BorderRadius.circular(18),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.08),
              blurRadius: 13,
              offset: const Offset(0, 5),
            ),
          ],
        ),
        child: Column(
          children: [
            Container(
              width: 39,
              height: 39,
              decoration: BoxDecoration(
                color: AppTheme.teal.withOpacity(0.12),
                borderRadius: BorderRadius.circular(13),
              ),
              child: Icon(icon, color: AppTheme.teal, size: 19),
            ),

            const SizedBox(height: 8),

            SizedBox(
              width: double.infinity,
              child: FittedBox(
                fit: BoxFit.scaleDown,
                child: Text(
                  title,
                  style: GoogleFonts.dmSans(
                    color: const Color(0xFF183636),
                    fontSize: 9.4,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // AI COMPLAINT ANALYSIS
  // ===========================================================================

  Widget _buildAiComplaintAnalysis() {
    return Container(
      padding: const EdgeInsets.all(17),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.52),
        borderRadius: BorderRadius.circular(23),
        border: Border.all(color: AppTheme.teal.withOpacity(0.65)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.16),
            blurRadius: 17,
            offset: const Offset(0, 7),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 45,
                height: 45,
                decoration: BoxDecoration(
                  color: AppTheme.teal,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: const Icon(
                  Icons.auto_awesome_rounded,
                  color: Colors.white,
                  size: 21,
                ),
              ),

              const SizedBox(width: 12),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'AI Complaint Analysis',
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 13,
                        fontWeight: FontWeight.w800,
                      ),
                    ),

                    const SizedBox(height: 3),

                    Text(
                      'Categorize complaints and get suggested actions.',
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.65),
                        fontSize: 9.7,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 16),

          SizedBox(
            width: double.infinity,
            height: 46,
            child: ElevatedButton.icon(
              // AI SCREEN ONLY
              onPressed: () {
                _pushScreen(
                  WardenAiComplaintScreen(hostelName: widget.hostelName),
                );
              },

              icon: const Icon(Icons.auto_awesome_rounded, size: 17),

              label: Text(
                'Analyze Complaint',
                style: GoogleFonts.dmSans(
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                ),
              ),

              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.teal,
                foregroundColor: Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(15),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // ATTENTION
  // ===========================================================================

  Widget _buildAttentionSection() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionHeader(
          title: 'Needs your attention',
          subtitle: 'Tasks requiring review',
          light: true,
        ),

        const SizedBox(height: 14),

        InkWell(
          borderRadius: BorderRadius.circular(22),

          // NORMAL COMPLAINTS LIST
          onTap: () {
            _pushScreen(WardenComplaintsScreen(hostelName: widget.hostelName));
          },

          child: Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(0.48),
              borderRadius: BorderRadius.circular(22),
              border: Border.all(
                color: const Color(0xFFFFA93A).withOpacity(0.55),
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 48,
                  height: 48,
                  decoration: BoxDecoration(
                    color: const Color(0xFFFFA93A).withOpacity(0.18),
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: const Icon(
                    Icons.warning_amber_rounded,
                    color: Color(0xFFFFA93A),
                    size: 22,
                  ),
                ),

                const SizedBox(width: 13),

                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '${_stats['complaints']} complaints need review',
                        style: GoogleFonts.dmSans(
                          color: Colors.white,
                          fontSize: 12.5,
                          fontWeight: FontWeight.w800,
                        ),
                      ),

                      const SizedBox(height: 4),

                      Text(
                        'Review pending complaints and resolve hostel issues.',
                        style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.64),
                          fontSize: 9.5,
                          height: 1.4,
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(width: 6),

                const Icon(
                  Icons.arrow_forward_rounded,
                  color: AppTheme.tealMint,
                  size: 19,
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // RECENT ACTIVITY
  // ===========================================================================

  Widget _buildRecentActivity() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.93),
        borderRadius: BorderRadius.circular(22),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.10),
            blurRadius: 16,
            offset: const Offset(0, 7),
          ),
        ],
      ),
      child: Column(
        children: _activity.asMap().entries.map((entry) {
          final item = entry.value;

          final last = entry.key == _activity.length - 1;

          return Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              border: last
                  ? null
                  : Border(
                      bottom: BorderSide(color: Colors.black.withOpacity(0.05)),
                    ),
            ),
            child: Row(
              children: [
                Container(
                  width: 39,
                  height: 39,
                  decoration: BoxDecoration(
                    color: AppTheme.teal.withOpacity(0.10),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    item['icon'] as IconData,
                    color: AppTheme.teal,
                    size: 18,
                  ),
                ),

                const SizedBox(width: 12),

                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        item['title'] as String,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          color: const Color(0xFF203A3A),
                          fontSize: 10.8,
                          fontWeight: FontWeight.w700,
                        ),
                      ),

                      const SizedBox(height: 2),

                      Text(
                        item['subtitle'] as String,
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          color: const Color(0xFF8F9A99),
                          fontSize: 9.2,
                        ),
                      ),
                    ],
                  ),
                ),

                Container(
                  width: 7,
                  height: 7,
                  decoration: const BoxDecoration(
                    color: AppTheme.teal,
                    shape: BoxShape.circle,
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  // ===========================================================================
  // DRAWER
  // ===========================================================================

  Widget _buildDrawer() {
    return Drawer(
      width: 305,
      backgroundColor: Colors.transparent,

      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.horizontal(right: Radius.circular(28)),
      ),

      child: VideoBackground(
        assetPath: 'assets/images/background.mp4',
        overlayOpacity: 0.62,
        cardMargin: EdgeInsets.zero,
        cardRadius: 0,

        child: SafeArea(
          child: Column(
            children: [
              // ===============================================================
              // PROFILE CARD
              // ===============================================================
              Container(
                width: double.infinity,
                margin: const EdgeInsets.all(14),
                padding: const EdgeInsets.all(17),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.48),
                  borderRadius: BorderRadius.circular(24),
                  border: Border.all(color: Colors.white.withOpacity(0.16)),
                ),
                child: Column(
                  children: [
                    Row(
                      children: [
                        Container(
                          width: 50,
                          height: 50,
                          decoration: const BoxDecoration(
                            color: AppTheme.teal,
                            shape: BoxShape.circle,
                          ),
                          child: Center(
                            child: Text(
                              _initial,
                              style: GoogleFonts.playfairDisplay(
                                color: Colors.white,
                                fontSize: 22,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(width: 12),

                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                widget.wardenName,
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: GoogleFonts.dmSans(
                                  color: Colors.white,
                                  fontSize: 12.5,
                                  fontWeight: FontWeight.w700,
                                ),
                              ),

                              const SizedBox(height: 2),

                              Text(
                                'Hostel Warden',
                                style: GoogleFonts.dmSans(
                                  color: Colors.white.withOpacity(0.60),
                                  fontSize: 9.5,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 15),

                    Container(
                      padding: const EdgeInsets.all(11),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.08),
                        borderRadius: BorderRadius.circular(13),
                      ),
                      child: Row(
                        children: [
                          const Icon(
                            Icons.apartment_rounded,
                            color: AppTheme.tealMint,
                            size: 16,
                          ),

                          const SizedBox(width: 8),

                          Expanded(
                            child: Text(
                              widget.hostelName,
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: GoogleFonts.dmSans(
                                color: Colors.white,
                                fontSize: 10.5,
                                fontWeight: FontWeight.w600,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              // ===============================================================
              // MENU
              // ===============================================================
              Expanded(
                child: SingleChildScrollView(
                  physics: const BouncingScrollPhysics(),

                  padding: const EdgeInsets.symmetric(horizontal: 12),

                  child: Column(
                    children: [
                      _drawerItem(
                        icon: Icons.dashboard_rounded,
                        title: 'Dashboard',
                        active: true,
                        onTap: () {
                          Navigator.pop(context);
                        },
                      ),

                      // -------------------------------------------------------
                      // NORMAL COMPLAINTS
                      // -------------------------------------------------------
                      _drawerItem(
                        icon: Icons.report_problem_outlined,
                        title: 'Complaints',
                        onTap: () {
                          _openDrawerScreen(
                            WardenComplaintsScreen(
                              hostelName: widget.hostelName,
                            ),
                          );
                        },
                      ),

                      _drawerItem(
                        icon: Icons.calendar_month_outlined,
                        title: 'Attendance',
                        onTap: () {
                          _openDrawerScreen(
                            WardenAttendanceScreen(
                              hostelName: widget.hostelName,
                            ),
                          );
                        },
                      ),

                      _drawerItem(
                        icon: Icons.qr_code_scanner_rounded,
                        title: 'Gate QR code',
                        onTap: () {
                          _openDrawerScreen(
                            WardenQrAttendanceScreen(
                              hostelName: widget.hostelName,
                            ),
                          );
                        },
                      ),

                      _drawerItem(
                        icon: Icons.campaign_outlined,
                        title: 'Announcements',
                        onTap: () {
                          _openDrawerScreen(
                            WardenAnnouncementsScreen(
                              hostelName: widget.hostelName,
                            ),
                          );
                        },
                      ),

                      _drawerItem(
                        icon: Icons.gavel_outlined,
                        title: 'Penalties',
                        onTap: () {
                          _openDrawerScreen(
                            WardenPenaltiesScreen(
                              hostelName: widget.hostelName,
                            ),
                          );
                        },
                      ),

                      _drawerItem(
                        icon: Icons.payments_outlined,
                        title: 'Fees & Rent',
                        onTap: () {
                          _openDrawerScreen(
                            WardenFeesScreen(hostelName: widget.hostelName),
                          );
                        },
                      ),

                      _drawerItem(
                        icon: Icons.meeting_room_outlined,
                        title: 'Room & Beds',
                        onTap: () {
                          _openDrawerScreen(
                            WardenRoomAllocationScreen(
                              hostelName: widget.hostelName,
                            ),
                          );
                        },
                      ),

                      _drawerItem(
                        icon: Icons.swap_horiz_rounded,
                        title: 'Room change requests',
                        onTap: () {
                          _openDrawerScreen(
                            WardenRoomChangeRequestsScreen(
                              hostelName: widget.hostelName,
                            ),
                          );
                        },
                      ),

                      _drawerItem(
                        icon: Icons.people_alt_outlined,
                        title: 'Students & Parents',
                        onTap: () {
                          _openDrawerScreen(
                            WardenStudentDetailsScreen(
                              hostelName: widget.hostelName,
                            ),
                          );
                        },
                      ),

                      _drawerItem(
                        icon: Icons.chat_bubble_outline_rounded,
                        title: 'Chat',
                        onTap: () {
                          _openDrawerScreen(
                            WardenChatScreen(hostelName: widget.hostelName),
                          );
                        },
                      ),

                      _drawerItem(
                        icon: Icons.cleaning_services_outlined,
                        title: 'Cleaning Schedule',
                        onTap: () {
                          _openDrawerScreen(
                            WardenCleaningScheduleScreen(
                              hostelName: widget.hostelName,
                            ),
                          );
                        },
                      ),

                      const SizedBox(height: 10),

                      _drawerSectionLabel('SMART FEATURE'),

                      // -------------------------------------------------------
                      // AI ONLY
                      // -------------------------------------------------------
                      _drawerItem(
                        icon: Icons.auto_awesome_rounded,
                        title: 'AI Complaint Analysis',
                        special: true,
                        onTap: () {
                          _openDrawerScreen(
                            WardenAiComplaintScreen(
                              hostelName: widget.hostelName,
                            ),
                          );
                        },
                      ),

                      _drawerItem(
                        icon: Icons.smart_toy_outlined,
                        title: 'AI Assistant',
                        special: true,
                        onTap: () {
                          _openDrawerScreen(const ChatbotScreen());
                        },
                      ),

                      const SizedBox(height: 8),
                    ],
                  ),
                ),
              ),

              // ===============================================================
              // LOGOUT
              // ===============================================================
              Padding(
                padding: const EdgeInsets.fromLTRB(14, 8, 14, 16),
                child: InkWell(
                  onTap: _logout,
                  borderRadius: BorderRadius.circular(16),
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(
                      horizontal: 15,
                      vertical: 13,
                    ),
                    decoration: BoxDecoration(
                      color: AppTheme.teal,
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: Row(
                      children: [
                        const Icon(
                          Icons.logout_rounded,
                          color: Colors.white,
                          size: 18,
                        ),

                        const SizedBox(width: 11),

                        Text(
                          'Logout',
                          style: GoogleFonts.dmSans(
                            color: Colors.white,
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // DRAWER COMPONENTS
  // ===========================================================================

  Widget _drawerSectionLabel(String text) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(13, 8, 13, 7),
      child: Align(
        alignment: Alignment.centerLeft,
        child: Text(
          text,
          style: GoogleFonts.dmSans(
            color: Colors.white.withOpacity(0.55),
            fontSize: 8,
            letterSpacing: 1.2,
            fontWeight: FontWeight.w800,
          ),
        ),
      ),
    );
  }

  Widget _drawerItem({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
    bool active = false,
    bool special = false,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 5),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(15),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 11),
          decoration: BoxDecoration(
            color: active || special
                ? AppTheme.teal.withOpacity(0.22)
                : Colors.black.withOpacity(0.20),
            borderRadius: BorderRadius.circular(15),
            border: Border.all(
              color: active || special
                  ? AppTheme.tealMint.withOpacity(0.30)
                  : Colors.white.withOpacity(0.07),
            ),
          ),
          child: Row(
            children: [
              Icon(
                icon,
                color: active || special
                    ? AppTheme.tealMint
                    : Colors.white.withOpacity(0.75),
                size: 19,
              ),

              const SizedBox(width: 12),

              Expanded(
                child: Text(
                  title,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: active || special ? AppTheme.tealMint : Colors.white,
                    fontSize: 10.8,
                    fontWeight: active || special
                        ? FontWeight.w700
                        : FontWeight.w500,
                  ),
                ),
              ),

              if (special)
                const Icon(
                  Icons.chevron_right_rounded,
                  color: AppTheme.tealMint,
                  size: 18,
                ),
            ],
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // SECTION HEADER
  // ===========================================================================

  Widget _sectionHeader({
    required String title,
    required String subtitle,
    bool light = false,
  }) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 4,
          height: 31,
          decoration: BoxDecoration(
            color: AppTheme.teal,
            borderRadius: BorderRadius.circular(3),
          ),
        ),

        const SizedBox(width: 10),

        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: GoogleFonts.dmSans(
                  color: light ? Colors.white : const Color(0xFF183636),
                  fontSize: 14.5,
                  fontWeight: FontWeight.w800,
                ),
              ),

              const SizedBox(height: 2),

              Text(
                subtitle,
                style: GoogleFonts.dmSans(
                  color: light
                      ? Colors.white.withOpacity(0.62)
                      : const Color(0xFF8B9797),
                  fontSize: 9.4,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
