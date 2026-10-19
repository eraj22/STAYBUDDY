import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme.dart';
import '../../../widgets/responsive_shell.dart';
import '../../../widgets/video_background.dart';
import '../../../api.dart';

import '../hostel/owner_hostel_screen.dart';
import '../wardens/owner_wardens_screen.dart';
import '../rooms/owner_rooms_screen.dart';
import '../students/owner_students_screen.dart';
import '../credentials/owner_credentials_screen.dart';
import '../reports/owner_reports_screen.dart';
import '../fees/owner_fees_screen.dart';
import '../../owner_login_screen.dart';
import '../../chatbot_screen.dart';

class OwnerDashboardScreen extends StatefulWidget {
  final String ownerName;
  final String ownerEmail;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;

  const OwnerDashboardScreen({
    super.key,
    required this.ownerName,
    required this.ownerEmail,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
  });

  @override
  State<OwnerDashboardScreen> createState() => _OwnerDashboardScreenState();
}

class _OwnerDashboardScreenState extends State<OwnerDashboardScreen> {
  static const _glassSurface = Color(0xE600383A);
  static const _glassBorder = Color(0x4D8AD7BF);
  static const _glassTextMuted = Color(0xFFD0E8E5);

  // ===========================================================================
  // REAL DASHBOARD DATA
  // ===========================================================================

  final Map<String, dynamic> _stats = {
    'students': 0,
    'wardens': 0,
    'complaints': 0,
    'revenue': 0,
    'occupied': 0,
    'total': 0,
    'available': 0,
    'pendingFees': 0,
  };

  final List<Map<String, dynamic>> _recentActivity = [];
  final GlobalKey<ScaffoldState> _dashboardScaffoldKey =
      GlobalKey<ScaffoldState>();

  bool _refreshing = false;

  @override
  void initState() {
    super.initState();
    _loadDashboard();
  }

  int _asInt(dynamic value) {
    if (value == null) return 0;

    if (value is int) return value;

    if (value is num) {
      return value.toInt();
    }

    return double.tryParse(value.toString())?.toInt() ?? 0;
  }

  List<Map<String, dynamic>> _activityFromDashboard(
    Map<String, dynamic> dashboard,
  ) {
    final source = dashboard['recent_activity'] ?? dashboard['activity'];
    final activities = <Map<String, dynamic>>[];

    if (source is List) {
      for (final rawItem in source.take(5)) {
        if (rawItem is! Map) continue;
        final item = Map<String, dynamic>.from(rawItem);
        activities.add({
          'icon': Icons.history_rounded,
          'color': AppTheme.tealMint,
          'title': item['title']?.toString() ?? 'Hostel activity updated',
          'sub':
              item['description']?.toString() ??
              item['message']?.toString() ??
              'A hostel update was recorded.',
          'time':
              item['time']?.toString() ??
              item['created_at']?.toString() ??
              'Recent',
        });
      }
    }

    if (activities.isNotEmpty) return activities;

    // The dashboard API does not always provide an event feed. In that case,
    // show a truthful snapshot derived from the live dashboard numbers.
    final students = _asInt(dashboard['students']);
    final wardens = _asInt(dashboard['wardens']);
    final complaints = _asInt(dashboard['complaints']);
    final available = _asInt(dashboard['available']);

    return [
      {
        'icon': Icons.sync_rounded,
        'color': AppTheme.tealMint,
        'title': 'Hostel overview refreshed',
        'sub': '$students student${students == 1 ? '' : 's'} in your hostel',
        'time': 'Now',
      },
      {
        'icon': Icons.bed_rounded,
        'color': const Color(0xFF9DD6FF),
        'title': '$available bed${available == 1 ? '' : 's'} available',
        'sub': 'Current room availability',
        'time': 'Live',
      },
      {
        'icon': complaints > 0
            ? Icons.report_problem_rounded
            : Icons.verified_rounded,
        'color': complaints > 0
            ? const Color(0xFFFFC978)
            : const Color(0xFFB9E6A1),
        'title': complaints > 0
            ? '$complaints complaint${complaints == 1 ? '' : 's'} need attention'
            : 'No pending complaints',
        'sub': '$wardens warden${wardens == 1 ? '' : 's'} currently assigned',
        'time': 'Live',
      },
    ];
  }

  Future<void> _loadDashboard() async {
    try {
      final response = await Api().getOwnerDashboard();

      final hostelsRaw = response['hostels'];

      if (hostelsRaw is! List || hostelsRaw.isEmpty) {
        throw Exception('No hostel dashboard data found');
      }

      Map<String, dynamic>? dashboard;

      for (final item in hostelsRaw) {
        if (item is Map) {
          final map = Map<String, dynamic>.from(item);

          if (map['hostel_name']?.toString().trim().toLowerCase() ==
              widget.hostelName.trim().toLowerCase()) {
            dashboard = map;
            break;
          }
        }
      }

      dashboard ??= Map<String, dynamic>.from(hostelsRaw.first as Map);

      if (!mounted) return;

      setState(() {
        _stats['students'] = _asInt(dashboard!['students']);

        _stats['wardens'] = _asInt(dashboard['wardens']);

        _stats['complaints'] = _asInt(dashboard['complaints']);

        _stats['revenue'] = _asInt(dashboard['revenue']);

        _stats['occupied'] = _asInt(dashboard['occupied']);

        _stats['total'] = _asInt(dashboard['total']);

        _stats['available'] = _asInt(dashboard['available']);

        _stats['pendingFees'] = _asInt(dashboard['pending_fees']);

        _recentActivity
          ..clear()
          ..addAll(_activityFromDashboard(dashboard));
      });
    } catch (error) {
      debugPrint('Owner dashboard load failed: $error');

      if (!mounted) return;

      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          SnackBar(content: Text('Could not load dashboard data.')),
        );
    }
  }

  // ===========================================================================
  // GETTERS
  // ===========================================================================

  int get _occupied => (_stats['occupied'] as num?)?.toInt() ?? 0;

  int get _total => (_stats['total'] as num?)?.toInt() ?? 0;

  int get _available => (_stats['available'] as num?)?.toInt() ?? 0;

  double get _occupancy {
    if (_total == 0) {
      return 0.0;
    }

    return (_occupied / _total).clamp(0.0, 1.0).toDouble();
  }

  int get _occupancyPercent => (_occupancy * 100).round();

  String get _ownerInitial {
    final name = widget.ownerName.trim();

    if (name.isEmpty) {
      return 'O';
    }

    return name[0].toUpperCase();
  }

  // ===========================================================================
  // REFRESH
  // ===========================================================================

  Future<void> _refreshDashboard() async {
    if (_refreshing) {
      return;
    }

    setState(() {
      _refreshing = true;
    });

    try {
      await _loadDashboard();
    } finally {
      if (!mounted) return;

      setState(() {
        _refreshing = false;
      });
    }
  }

  // ===========================================================================
  // NAVIGATION
  // ===========================================================================

  void _openHostelProfile() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerHostelScreen(
          ownerName: widget.ownerName,
          ownerEmail: widget.ownerEmail,
          hostelName: widget.hostelName,
          hostelType: widget.hostelType,
          city: widget.city,
          address: widget.address,
        ),
      ),
    );
  }

  void _openRooms() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerRoomsScreen(hostelName: widget.hostelName),
      ),
    );
  }

  void _openStudents() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerStudentsScreen(hostelName: widget.hostelName),
      ),
    );
  }

  void _openWardens() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerWardensScreen(hostelName: widget.hostelName),
      ),
    );
  }

  void _openCredentials() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerCredentialsScreen(hostelName: widget.hostelName),
      ),
    );
  }

  void _openReports() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerReportsScreen(hostelName: widget.hostelName),
      ),
    );
  }

  void _openFees() {
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerFeesScreen(hostelName: widget.hostelName),
      ),
    );
  }

  Future<void> _logout() async {
    await AuthStore.clear();
    if (!mounted) return;
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => const OwnerLoginScreen()),
      (route) => false,
    );
  }

  void _comingSoon(String title) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          margin: const EdgeInsets.all(18),
          backgroundColor: const Color(0xFF123F40),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
          ),
          content: Row(
            children: [
              const Icon(
                Icons.auto_awesome_rounded,
                color: Colors.white,
                size: 18,
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  '$title will be connected later.',
                  style: GoogleFonts.dmSans(
                    fontSize: 13,
                    fontWeight: FontWeight.w500,
                    color: Colors.white,
                  ),
                ),
              ),
            ],
          ),
        ),
      );
  }

  // ===========================================================================
  // OWNER MENU
  // ===========================================================================

  Widget _buildOwnerDrawer() {
    return Drawer(
      width: MediaQuery.of(context).size.width * 0.88,
      backgroundColor: Colors.transparent,
      elevation: 0,
      child: VideoBackground(
        assetPath: 'assets/images/background.mp4',
        overlayOpacity: 0.48,
        child: SafeArea(
          child: Container(
            margin: const EdgeInsets.fromLTRB(10, 10, 4, 10),
            padding: const EdgeInsets.fromLTRB(14, 16, 14, 18),
            decoration: BoxDecoration(
              color: const Color(0xE6003133),
              borderRadius: BorderRadius.circular(28),
              border: Border.all(color: _glassBorder),
            ),
            child: ListView(
              children: [
                _buildMenuProfileCard(),
                const SizedBox(height: 18),
                _drawerItem(
                  icon: Icons.dashboard_rounded,
                  title: 'Dashboard',
                  subtitle: 'Hostel overview',
                  selected: true,
                  onTap: () => Navigator.pop(context),
                ),
                _drawerItem(
                  icon: Icons.apartment_rounded,
                  title: 'Hostel Profile',
                  subtitle: 'Hostel, floors and details',
                  onTap: _openHostelProfile,
                ),
                _drawerItem(
                  icon: Icons.bed_rounded,
                  title: 'Rooms & Beds',
                  subtitle: 'Floors, rooms and bed status',
                  onTap: _openRooms,
                ),
                _drawerItem(
                  icon: Icons.groups_rounded,
                  title: 'Students',
                  subtitle: 'Student allocation and records',
                  onTap: _openStudents,
                ),
                _drawerItem(
                  icon: Icons.receipt_long_rounded,
                  title: 'Fees & Rent',
                  subtitle: 'Monthly rent and payment records',
                  onTap: _openFees,
                ),
                _drawerItem(
                  icon: Icons.badge_outlined,
                  title: 'Manage Wardens',
                  subtitle: 'Add and manage warden accounts',
                  onTap: _openWardens,
                ),
                _drawerItem(
                  icon: Icons.password_rounded,
                  title: 'Login Credentials',
                  subtitle: 'Manage generated credentials',
                  onTap: _openCredentials,
                ),
                _drawerItem(
                  icon: Icons.insights_rounded,
                  title: 'Reports',
                  subtitle: 'Occupancy and hostel insights',
                  onTap: _openReports,
                ),
                _drawerItem(
                  icon: Icons.smart_toy_outlined,
                  title: 'AI Assistant',
                  subtitle: 'Ask about your hostels and operations',
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(builder: (_) => const ChatbotScreen()),
                    );
                  },
                ),
                const SizedBox(height: 8),
                _drawerItem(
                  icon: Icons.logout_rounded,
                  title: 'Log out',
                  subtitle: 'Sign out from this owner account',
                  onTap: _logout,
                  danger: true,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _drawerItem({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
    bool selected = false,
    bool danger = false,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 9),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: () {
            Navigator.pop(context);
            if (!selected) Future.microtask(onTap);
          },
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 13, vertical: 11),
            decoration: BoxDecoration(
              color: selected
                  ? AppTheme.teal.withValues(alpha: 0.26)
                  : Colors.black.withValues(alpha: 0.16),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(
                color: selected
                    ? AppTheme.tealMint.withValues(alpha: 0.55)
                    : Colors.white.withValues(alpha: 0.10),
              ),
            ),
            child: Row(
              children: [
                Icon(
                  icon,
                  color: danger
                      ? const Color(0xFFFFA3A3)
                      : (selected ? AppTheme.tealMint : Colors.white),
                  size: 20,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        style: GoogleFonts.dmSans(
                          color: danger
                              ? const Color(0xFFFFD1D1)
                              : Colors.white,
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        subtitle,
                        style: GoogleFonts.dmSans(
                          color: Colors.white.withValues(alpha: 0.58),
                          fontSize: 8.5,
                        ),
                      ),
                    ],
                  ),
                ),
                Icon(
                  selected ? Icons.check_rounded : Icons.chevron_right_rounded,
                  color: danger
                      ? const Color(0xFFFFA3A3)
                      : (selected ? AppTheme.tealMint : Colors.white54),
                  size: 18,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildMenuProfileCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.tealMint.withValues(alpha: 0.12),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _glassBorder),
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: const BoxDecoration(
              color: AppTheme.teal,
              shape: BoxShape.circle,
            ),
            child: Center(
              child: Text(
                _ownerInitial,
                style: GoogleFonts.playfairDisplay(
                  color: Colors.white,
                  fontSize: 21,
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
                  widget.ownerName,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  widget.ownerEmail,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: _glassTextMuted,
                    fontSize: 10.5,
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
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        key: _dashboardScaffoldKey,
        backgroundColor: Colors.transparent,
        drawer: _buildOwnerDrawer(),
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.46,
          child: RefreshIndicator(
            color: AppTheme.tealMint,
            onRefresh: _refreshDashboard,
            child: CustomScrollView(
              physics: const AlwaysScrollableScrollPhysics(
                parent: BouncingScrollPhysics(),
              ),
              slivers: [
                SliverToBoxAdapter(child: _buildHeader()),

                SliverPadding(
                  padding: const EdgeInsets.fromLTRB(18, 20, 18, 0),
                  sliver: SliverToBoxAdapter(child: _buildOccupancyCard()),
                ),

                SliverPadding(
                  padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                  sliver: SliverToBoxAdapter(
                    child: _sectionHeader(
                      title: 'Hostel Overview',
                      subtitle: 'Today at a glance',
                    ),
                  ),
                ),

                SliverPadding(
                  padding: const EdgeInsets.symmetric(horizontal: 18),
                  sliver: SliverToBoxAdapter(child: _buildStats()),
                ),

                const SliverToBoxAdapter(child: SizedBox(height: 36)),
              ],
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
    return Container(
      decoration: const BoxDecoration(
        color: Color(0xA6003032),
        borderRadius: BorderRadius.vertical(bottom: Radius.circular(34)),
        border: Border(bottom: BorderSide(color: Color(0x338AD7BF))),
      ),
      child: Stack(
        children: [
          Positioned(
            top: -70,
            right: -65,
            child: Container(
              width: 190,
              height: 190,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.white.withOpacity(0.045),
              ),
            ),
          ),

          Positioned(
            top: 120,
            left: -80,
            child: Container(
              width: 180,
              height: 180,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: AppTheme.tealMint.withOpacity(0.05),
              ),
            ),
          ),

          SafeArea(
            bottom: false,
            child: Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 26),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildTopBar(),

                  const SizedBox(height: 22),

                  Text(
                    'Your hostel at a glance',
                    style: GoogleFonts.dmSans(
                      fontSize: 13,
                      color: Colors.white.withOpacity(0.68),
                    ),
                  ),

                  const SizedBox(height: 5),

                  Text(
                    'Everything looks good today.',
                    style: GoogleFonts.playfairDisplay(
                      fontSize: 25,
                      height: 1.15,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                  ),

                  const SizedBox(height: 20),

                  _buildHostelCard(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // TOP BAR
  // ===========================================================================

  Widget _buildTopBar() {
    return Row(
      children: [
        Container(
          width: 50,
          height: 50,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.13),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white.withOpacity(0.20)),
          ),
          child: Center(
            child: Text(
              _ownerInitial,
              style: GoogleFonts.playfairDisplay(
                color: Colors.white,
                fontSize: 21,
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
                'Welcome back,',
                style: GoogleFonts.dmSans(
                  color: Colors.white.withOpacity(0.64),
                  fontSize: 12,
                  fontWeight: FontWeight.w400,
                ),
              ),

              const SizedBox(height: 1),

              Text(
                widget.ownerName,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.playfairDisplay(
                  fontSize: 20,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(width: 12),

        // =========================================================
        // ONLY MENU ICON
        // SEARCH + BELL REMOVED
        // =========================================================
        _menuButton(),
      ],
    );
  }

  Widget _menuButton() {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () => _dashboardScaffoldKey.currentState?.openDrawer(),
        borderRadius: BorderRadius.circular(14),
        child: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.12),
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: Colors.white.withOpacity(0.15)),
          ),
          child: const Icon(Icons.menu_rounded, color: Colors.white, size: 23),
        ),
      ),
    );
  }

  // ===========================================================================
  // HOSTEL CARD
  // ===========================================================================

  Widget _buildHostelCard() {
    final typeText = widget.hostelType.trim().isEmpty
        ? 'Hostel'
        : widget.hostelType;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: _openHostelProfile,
        borderRadius: BorderRadius.circular(20),
        child: Container(
          padding: const EdgeInsets.all(15),
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.11),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white.withOpacity(0.17)),
          ),
          child: Row(
            children: [
              Container(
                width: 47,
                height: 47,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.14),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: const Icon(
                  Icons.apartment_rounded,
                  color: Colors.white,
                  size: 23,
                ),
              ),

              const SizedBox(width: 13),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.hostelName,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),

                    const SizedBox(height: 4),

                    Row(
                      children: [
                        Icon(
                          Icons.location_on_outlined,
                          size: 13,
                          color: Colors.white.withOpacity(0.60),
                        ),

                        const SizedBox(width: 3),

                        Flexible(
                          child: Text(
                            '${widget.city} · $typeText',
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.dmSans(
                              fontSize: 11,
                              color: Colors.white.withOpacity(0.62),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),

              const SizedBox(width: 10),

              Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 6),
                decoration: BoxDecoration(
                  color: AppTheme.tealMint.withOpacity(0.13),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 7,
                      height: 7,
                      decoration: const BoxDecoration(
                        color: AppTheme.tealMint,
                        shape: BoxShape.circle,
                      ),
                    ),

                    const SizedBox(width: 5),

                    Text(
                      'Active',
                      style: GoogleFonts.dmSans(
                        color: AppTheme.tealMint,
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // OCCUPANCY
  // ===========================================================================

  Widget _buildOccupancyCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: _glassSurface,
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: _glassBorder, width: 0.7),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.035),
            blurRadius: 20,
            offset: const Offset(0, 8),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Occupancy Overview',
                      style: GoogleFonts.dmSans(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),

                    const SizedBox(height: 4),

                    Text(
                      'Current bed utilization',
                      style: GoogleFonts.dmSans(
                        fontSize: 11,
                        color: _glassTextMuted,
                      ),
                    ),
                  ],
                ),
              ),

              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: AppTheme.tealMint.withValues(alpha: 0.16),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '$_occupancyPercent%',
                  style: GoogleFonts.dmSans(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.tealMint,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 18),

          ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: LinearProgressIndicator(
              value: _occupancy,
              minHeight: 10,
              backgroundColor: Colors.white.withValues(alpha: 0.14),
              valueColor: const AlwaysStoppedAnimation<Color>(
                AppTheme.tealMint,
              ),
            ),
          ),

          const SizedBox(height: 17),

          Row(
            children: [
              Expanded(
                child: _occupancyInfo(
                  value: '$_occupied',
                  label: 'Occupied',
                  color: AppTheme.tealMint,
                ),
              ),

              Container(height: 38, width: 1, color: _glassBorder),

              Expanded(
                child: _occupancyInfo(
                  value: '$_available',
                  label: 'Available',
                  color: AppTheme.tealMint,
                ),
              ),

              Container(height: 38, width: 1, color: _glassBorder),

              Expanded(
                child: _occupancyInfo(
                  value: '$_total',
                  label: 'Total beds',
                  color: Colors.white,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _occupancyInfo({
    required String value,
    required String label,
    required Color color,
  }) {
    return Column(
      children: [
        Text(
          value,
          style: GoogleFonts.dmSans(
            fontSize: 18,
            fontWeight: FontWeight.w700,
            color: color,
          ),
        ),

        const SizedBox(height: 2),

        Text(
          label,
          style: GoogleFonts.dmSans(fontSize: 10, color: _glassTextMuted),
        ),
      ],
    );
  }

  // ===========================================================================
  // STATS
  // ===========================================================================

  Widget _buildStats() {
    final stats = [
      _DashboardStat(
        value: '${_stats['students']}',
        label: 'Students',
        icon: Icons.school_rounded,
        color: AppTheme.tealMint,
        background: const Color(0x332A9F8F),
      ),
      _DashboardStat(
        value: '${_stats['wardens']}',
        label: 'Wardens',
        icon: Icons.badge_rounded,
        color: const Color(0xFF9DD6FF),
        background: const Color(0x332E7FB4),
      ),
      _DashboardStat(
        value: '${_stats['complaints']}',
        label: 'Complaints',
        icon: Icons.report_problem_rounded,
        color: const Color(0xFFFFC978),
        background: const Color(0x334F3919),
      ),
      _DashboardStat(
        value: '${_stats['pendingFees']}',
        label: 'Pending Fees',
        icon: Icons.receipt_long_rounded,
        color: const Color(0xFFD7B8FF),
        background: const Color(0x333E2362),
      ),
    ];

    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;

        final cardWidth = (width - 12) / 2;

        return Wrap(
          spacing: 12,
          runSpacing: 12,
          children: stats.map((item) {
            return SizedBox(width: cardWidth, child: _statCard(item));
          }).toList(),
        );
      },
    );
  }

  Widget _statCard(_DashboardStat item) {
    return Container(
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: _glassSurface,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: _glassBorder, width: 0.6),
      ),
      child: Row(
        children: [
          Container(
            width: 43,
            height: 43,
            decoration: BoxDecoration(
              color: item.background,
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(item.icon, size: 20, color: item.color),
          ),

          const SizedBox(width: 12),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  item.value,
                  style: GoogleFonts.dmSans(
                    fontSize: 18,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),

                const SizedBox(height: 2),

                Text(
                  item.label,
                  maxLines: 2,
                  softWrap: true,
                  overflow: TextOverflow.visible,
                  style: GoogleFonts.dmSans(
                    fontSize: 10.5,
                    color: _glassTextMuted,
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
  // REVENUE
  // ===========================================================================

  Widget _buildRevenueCard() {
    final revenue = (_stats['revenue'] as num?)?.toInt() ?? 0;

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: _openReports,
        borderRadius: BorderRadius.circular(24),
        child: Container(
          padding: const EdgeInsets.all(20),
          decoration: BoxDecoration(
            color: const Color(0xFF103F40),
            borderRadius: BorderRadius.circular(24),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF103F40).withOpacity(0.14),
                blurRadius: 22,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Stack(
            children: [
              Positioned(
                right: -40,
                top: -55,
                child: Container(
                  width: 130,
                  height: 130,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.04),
                    shape: BoxShape.circle,
                  ),
                ),
              ),

              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 43,
                        height: 43,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.10),
                          borderRadius: BorderRadius.circular(13),
                        ),
                        child: const Icon(
                          Icons.account_balance_wallet_rounded,
                          color: AppTheme.tealMint,
                          size: 21,
                        ),
                      ),

                      const SizedBox(width: 12),

                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Monthly Revenue',
                              style: GoogleFonts.dmSans(
                                fontSize: 12,
                                color: Colors.white.withOpacity(0.65),
                              ),
                            ),

                            const SizedBox(height: 2),

                            Text(
                              'PKR ${_formatAmount(revenue)}',
                              style: GoogleFonts.dmSans(
                                fontSize: 21,
                                fontWeight: FontWeight.w700,
                                color: Colors.white,
                              ),
                            ),
                          ],
                        ),
                      ),

                      const Icon(
                        Icons.arrow_forward_ios_rounded,
                        size: 15,
                        color: Colors.white54,
                      ),
                    ],
                  ),

                  const SizedBox(height: 19),

                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 9,
                          vertical: 6,
                        ),
                        decoration: BoxDecoration(
                          color: AppTheme.tealMint.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Row(
                          children: [
                            const Icon(
                              Icons.trending_up_rounded,
                              size: 14,
                              color: AppTheme.tealMint,
                            ),

                            const SizedBox(width: 4),

                            Text(
                              '8.4%',
                              style: GoogleFonts.dmSans(
                                fontSize: 10,
                                fontWeight: FontWeight.w700,
                                color: AppTheme.tealMint,
                              ),
                            ),
                          ],
                        ),
                      ),

                      const SizedBox(width: 8),

                      Text(
                        'from last month',
                        style: GoogleFonts.dmSans(
                          fontSize: 10.5,
                          color: Colors.white.withOpacity(0.50),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // ACTIVITY
  // ===========================================================================

  Widget _buildActivity() {
    return Container(
      decoration: BoxDecoration(
        color: _glassSurface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: _glassBorder, width: 0.6),
      ),
      child: Column(
        children: List.generate(_recentActivity.length, (index) {
          final item = _recentActivity[index];

          return Column(
            children: [
              _activityItem(item),

              if (index < _recentActivity.length - 1)
                Padding(
                  padding: const EdgeInsets.only(left: 68),
                  child: Divider(
                    height: 1,
                    thickness: 0.7,
                    color: _glassBorder,
                  ),
                ),
            ],
          );
        }),
      ),
    );
  }

  Widget _activityItem(Map<String, dynamic> item) {
    final color = item['color'] as Color;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 13),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              color: color.withOpacity(0.09),
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(item['icon'] as IconData, size: 19, color: color),
          ),

          const SizedBox(width: 12),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  item['title'].toString(),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: Colors.white,
                  ),
                ),

                const SizedBox(height: 3),

                Text(
                  item['sub'].toString(),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 10.5,
                    color: _glassTextMuted,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(width: 8),

          Text(
            item['time'].toString(),
            style: GoogleFonts.dmSans(fontSize: 9.5, color: _glassTextMuted),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // SECTION HEADER
  // ===========================================================================

  Widget _sectionHeader({
    required String title,
    String? subtitle,
    String? trailing,
    VoidCallback? onTrailingTap,
  }) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.end,
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: GoogleFonts.playfairDisplay(
                  fontSize: 19,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),

              if (subtitle != null) ...[
                const SizedBox(height: 2),

                Text(
                  subtitle,
                  style: GoogleFonts.dmSans(
                    fontSize: 10.5,
                    color: Colors.white70,
                  ),
                ),
              ],
            ],
          ),
        ),

        if (trailing != null)
          InkWell(
            onTap: onTrailingTap,
            borderRadius: BorderRadius.circular(10),
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 4),
              child: Row(
                children: [
                  Text(
                    trailing,
                    style: GoogleFonts.dmSans(
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      color: AppTheme.teal,
                    ),
                  ),

                  const SizedBox(width: 3),

                  const Icon(
                    Icons.arrow_forward_rounded,
                    size: 14,
                    color: AppTheme.teal,
                  ),
                ],
              ),
            ),
          ),
      ],
    );
  }

  // ===========================================================================
  // HELPERS
  // ===========================================================================

  String _formatAmount(int amount) {
    final value = amount.toString();

    if (value.length <= 3) {
      return value;
    }

    final buffer = StringBuffer();

    for (int i = 0; i < value.length; i++) {
      final remaining = value.length - i;

      buffer.write(value[i]);

      if (remaining > 1 && remaining % 3 == 1) {
        buffer.write(',');
      }
    }

    return buffer.toString();
  }
}

// =============================================================================
// DASHBOARD MODELS
// =============================================================================

class _DashboardStat {
  final String value;
  final String label;
  final IconData icon;
  final Color color;
  final Color background;

  const _DashboardStat({
    required this.value,
    required this.label,
    required this.icon,
    required this.color,
    required this.background,
  });
}
