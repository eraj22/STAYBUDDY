import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
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
    with TickerProviderStateMixin {

  Map<String, dynamic> _stats = {
    'students': 62, 'complaints': 4, 'attendance': 89, 'available': 18
  };
  bool _loadingStats = true;

  late final AnimationController _headerAnim;
  late final AnimationController _gridAnim;
  late final Animation<double> _headerFade;
  late final Animation<double> _headerSlide;

  final _activity = [
    {'icon': Icons.report_rounded, 'color': Color(0xFFB85C00),
     'title': 'New complaint — WiFi issue', 'sub': 'Sara Khan · Room 102 · 2 hrs ago', 'dot': Color(0xFFFF8C00)},
    {'icon': Icons.login_rounded, 'color': Color(0xFF0A6B6E),
     'title': 'Student checked in', 'sub': 'Amna Raza · Room 201 · 5 hrs ago', 'dot': AppTheme.teal},
    {'icon': Icons.bed_rounded, 'color': Color(0xFF2D6A11),
     'title': 'Bed vacancy — Room 301', 'sub': '1 bed available · Yesterday', 'dot': Color(0xFF4CAF50)},
    {'icon': Icons.campaign_rounded, 'color': Color(0xFF185FA5),
     'title': 'Announcement posted', 'sub': 'Mess timing update · Yesterday', 'dot': Color(0xFF185FA5)},
    {'icon': Icons.auto_awesome_rounded, 'color': AppTheme.teal,
     'title': 'AI categorized 3 complaints', 'sub': 'Maintenance ×2, Cleanliness ×1', 'dot': AppTheme.teal},
  ];

  List<Map<String, dynamic>> get _actions => [
    {'label': 'Complaints', 'sublabel': '+ AI', 'icon': Icons.report_rounded,
     'gradient': [Color(0xFFFF8C00), Color(0xFFB85C00)],
     'badge': '${_stats['complaints']}',
     'screen': WardenComplaintsScreen(hostelName: widget.hostelName)},
    {'label': 'Attendance', 'sublabel': 'Daily', 'icon': Icons.calendar_month_rounded,
     'gradient': [Color(0xFF0A8B8E), Color(0xFF063E40)],
     'badge': null,
     'screen': WardenAttendanceScreen(hostelName: widget.hostelName)},
    {'label': 'Announce', 'sublabel': 'Broadcast', 'icon': Icons.campaign_rounded,
     'gradient': [Color(0xFF43A047), Color(0xFF2D6A11)],
     'badge': null,
     'screen': WardenAnnouncementsScreen(hostelName: widget.hostelName)},
    {'label': 'Penalties', 'sublabel': 'Rules', 'icon': Icons.gavel_rounded,
     'gradient': [Color(0xFFE53935), Color(0xFFC62828)],
     'badge': null,
     'screen': WardenPenaltiesScreen(hostelName: widget.hostelName)},
    {'label': 'Rooms', 'sublabel': 'Allocate', 'icon': Icons.meeting_room_rounded,
     'gradient': [Color(0xFF9C27B0), Color(0xFF7B3FC4)],
     'badge': null,
     'screen': WardenRoomAllocationScreen(hostelName: widget.hostelName)},
    {'label': 'Students', 'sublabel': 'Details', 'icon': Icons.people_rounded,
     'gradient': [Color(0xFF1976D2), Color(0xFF185FA5)],
     'badge': null,
     'screen': WardenStudentDetailsScreen(hostelName: widget.hostelName)},
    {'label': 'Empty Beds', 'sublabel': 'Vacancies', 'icon': Icons.bed_rounded,
     'gradient': [Color(0xFF00897B), Color(0xFF00695C)],
     'badge': '3',
     'screen': WardenEmptyBedsScreen(hostelName: widget.hostelName)},
    {'label': 'Chat', 'sublabel': 'Parents', 'icon': Icons.chat_bubble_rounded,
     'gradient': [Color(0xFF8E24AA), Color(0xFF6A1B9A)],
     'badge': '3',
     'screen': WardenChatScreen(hostelName: widget.hostelName)},
    {'label': 'Cleaning', 'sublabel': 'Schedule', 'icon': Icons.cleaning_services_rounded,
     'gradient': [Color(0xFF00ACC1), Color(0xFF0A6B6E)],
     'badge': null,
     'screen': WardenCleaningScheduleScreen(hostelName: widget.hostelName)},
  ];

  @override
  void initState() {
    super.initState();
    _headerAnim = AnimationController(vsync: this,
        duration: const Duration(milliseconds: 900));
    _gridAnim = AnimationController(vsync: this,
        duration: const Duration(milliseconds: 600));
    _headerFade = CurvedAnimation(parent: _headerAnim, curve: Curves.easeOut);
    _headerSlide = Tween<double>(begin: -30, end: 0).animate(
        CurvedAnimation(parent: _headerAnim, curve: Curves.easeOutCubic));
    _headerAnim.forward();
    Future.delayed(const Duration(milliseconds: 300), () {
      if (mounted) _gridAnim.forward();
    });
    _loadDashboard();
  }

  Future<void> _loadDashboard() async {
    try {
      final data = await Api().getWardenDashboard();
      if (!mounted) return;
      setState(() {
        _stats = {
          'students':   data['students'] ?? 62,
          'complaints': data['complaints'] ?? 4,
          'attendance': 89,
          'available':  data['available_beds'] ?? 18,
        };
        _loadingStats = false;
      });
    } catch (_) {
      if (mounted) setState(() => _loadingStats = false);
    }
  }

  @override
  void dispose() {
    _headerAnim.dispose();
    _gridAnim.dispose();
    super.dispose();
  }

  String get _greeting {
    final h = DateTime.now().hour;
    if (h < 12) return 'Good Morning';
    if (h < 17) return 'Good Afternoon';
    return 'Good Evening';
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: const Color(0xFFF0F4F4),
        body: CustomScrollView(
          slivers: [
            // ── Premium Header ──────────────────────────────────────
            SliverToBoxAdapter(child: _buildHeader()),

            // ── Stats row ──────────────────────────────────────────
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 0),
              sliver: SliverToBoxAdapter(child: _buildStats())),

            // ── Section title ──────────────────────────────────────
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(20, 26, 20, 12),
              sliver: SliverToBoxAdapter(
                child: Row(children: [
                  Container(width: 3, height: 18,
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFF0A8B8E), Color(0xFF4DD9C0)],
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter),
                      borderRadius: BorderRadius.circular(2))),
                  const SizedBox(width: 10),
                  Text('Management Tools',
                    style: GoogleFonts.dmSans(
                      fontSize: 15, fontWeight: FontWeight.w700,
                      color: const Color(0xFF0D1F1F))),
                  const Spacer(),
                  Text('${_actions.length} modules',
                    style: GoogleFonts.dmSans(
                      fontSize: 11, color: Colors.black38)),
                ]),
              )),

            // ── Action grid ────────────────────────────────────────
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 0),
              sliver: SliverToBoxAdapter(child: _buildGrid())),

            // ── Activity section ───────────────────────────────────
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(20, 26, 20, 12),
              sliver: SliverToBoxAdapter(
                child: Row(children: [
                  Container(width: 3, height: 18,
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(
                        colors: [Color(0xFFFF8C00), Color(0xFFFFCC02)],
                        begin: Alignment.topCenter, end: Alignment.bottomCenter),
                      borderRadius: BorderRadius.circular(2))),
                  const SizedBox(width: 10),
                  Text('Recent Activity',
                    style: GoogleFonts.dmSans(
                      fontSize: 15, fontWeight: FontWeight.w700,
                      color: const Color(0xFF0D1F1F))),
                ]),
              )),

            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 40),
              sliver: SliverToBoxAdapter(child: _buildActivity())),
          ],
        ),
      ),
    );
  }

  // ── Header ──────────────────────────────────────────────────────────────────
  Widget _buildHeader() {
    return AnimatedBuilder(
      animation: _headerAnim,
      builder: (_, child) => Transform.translate(
        offset: Offset(0, _headerSlide.value),
        child: Opacity(opacity: _headerFade.value, child: child)),
      child: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [Color(0xFF041E1F), Color(0xFF063E40), Color(0xFF0A6B6E)],
            stops: [0.0, 0.5, 1.0])),
        child: SafeArea(
          bottom: false,
          child: Stack(children: [
            // Background decorative circles
            Positioned(top: -20, right: -20,
              child: Container(width: 130, height: 130,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: RadialGradient(colors: [
                    const Color(0xFF4DD9C0).withOpacity(0.12),
                    Colors.transparent])))),
            Positioned(bottom: 0, left: -30,
              child: Container(width: 100, height: 100,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: RadialGradient(colors: [
                    Colors.white.withOpacity(0.05),
                    Colors.transparent])))),

            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Top row
                  Row(children: [
                    // Avatar
                    Container(
                      width: 50, height: 50,
                      decoration: BoxDecoration(
                        gradient: const LinearGradient(
                          colors: [Color(0xFF4DD9C0), Color(0xFF0A8B8E)]),
                        shape: BoxShape.circle,
                        boxShadow: [BoxShadow(
                          color: const Color(0xFF4DD9C0).withOpacity(0.3),
                          blurRadius: 12, offset: const Offset(0, 4))]),
                      child: Center(child: Text(
                        widget.wardenName.isNotEmpty
                            ? widget.wardenName[0].toUpperCase() : 'W',
                        style: GoogleFonts.cormorantGaramond(
                          fontSize: 22, fontWeight: FontWeight.w700,
                          color: Colors.white)))),
                    const SizedBox(width: 13),
                    Expanded(child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(_greeting,
                          style: GoogleFonts.dmSans(
                            color: Colors.white.withOpacity(0.5),
                            fontSize: 11, letterSpacing: 0.5)),
                        Text(widget.wardenName,
                          style: GoogleFonts.cormorantGaramond(
                            fontSize: 22, fontWeight: FontWeight.w700,
                            color: Colors.white, letterSpacing: 0.3)),
                      ])),
                    // Logout
                    GestureDetector(
                      onTap: () async {
                        await AuthStore.clear();
                        if (mounted) Navigator.popUntil(context, (r) => r.isFirst);
                      },
                      child: Container(
                        width: 38, height: 38,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.08),
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white.withOpacity(0.15))),
                        child: const Icon(Icons.logout_rounded,
                            color: Colors.white, size: 16))),
                  ]),

                  const SizedBox(height: 16),

                  // Hostel info pill
                  Container(
                    padding: const EdgeInsets.fromLTRB(14, 11, 14, 11),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.07),
                      borderRadius: BorderRadius.circular(14),
                      border: Border.all(color: Colors.white.withOpacity(0.12))),
                    child: Row(children: [
                      Container(
                        width: 32, height: 32,
                        decoration: BoxDecoration(
                          color: const Color(0xFF4DD9C0).withOpacity(0.15),
                          shape: BoxShape.circle),
                        child: const Icon(Icons.apartment_rounded,
                            color: Color(0xFF4DD9C0), size: 16)),
                      const SizedBox(width: 10),
                      Expanded(child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('Managing',
                            style: GoogleFonts.dmSans(
                              color: Colors.white.withOpacity(0.4),
                              fontSize: 10, letterSpacing: 0.5)),
                          Text(widget.hostelName,
                            style: GoogleFonts.dmSans(
                              color: Colors.white,
                              fontSize: 13, fontWeight: FontWeight.w600)),
                        ])),
                      Container(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: const Color(0xFF4DD9C0).withOpacity(0.15),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                              color: const Color(0xFF4DD9C0).withOpacity(0.3))),
                        child: Row(mainAxisSize: MainAxisSize.min, children: [
                          Container(width: 5, height: 5,
                            decoration: const BoxDecoration(
                              color: Color(0xFF4DD9C0),
                              shape: BoxShape.circle)),
                          const SizedBox(width: 5),
                          Text('Active',
                            style: GoogleFonts.dmSans(
                              color: const Color(0xFF4DD9C0),
                              fontSize: 10, fontWeight: FontWeight.w700)),
                        ])),
                    ]),
                  ),
                  const SizedBox(height: 20),
                ],
              ),
            ),
          ]),
        ),
      ),
    );
  }

  // ── Stats ────────────────────────────────────────────────────────────────────
  Widget _buildStats() {
    final items = [
      {'val': '${_stats['students']}', 'label': 'Students',
       'icon': Icons.school_rounded,
       'color': const Color(0xFF0A8B8E), 'bg': const Color(0xFFE0F5F5)},
      {'val': '${_stats['complaints']}', 'label': 'Issues',
       'icon': Icons.report_rounded,
       'color': const Color(0xFFB85C00), 'bg': const Color(0xFFFEF3DF)},
      {'val': '${_stats['attendance']}%', 'label': 'Attend.',
       'icon': Icons.check_circle_outline_rounded,
       'color': const Color(0xFF2D6A11), 'bg': const Color(0xFFEAF5E3)},
      {'val': '${_stats['available']}', 'label': 'Free Beds',
       'icon': Icons.bed_rounded,
       'color': const Color(0xFF185FA5), 'bg': const Color(0xFFE3EEF9)},
    ];

    return Transform.translate(
      offset: const Offset(0, -20),
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 0),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(22),
          boxShadow: [
            BoxShadow(color: Colors.black.withOpacity(0.08),
                blurRadius: 20, offset: const Offset(0, 4))
          ]),
        child: Row(
          children: items.asMap().entries.map((e) {
            final item = e.value;
            return Expanded(
              child: Container(
                margin: EdgeInsets.only(right: e.key < items.length - 1 ? 10 : 0),
                padding: const EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: item['bg'] as Color,
                  borderRadius: BorderRadius.circular(14)),
                child: Column(children: [
                  Icon(item['icon'] as IconData,
                      size: 18, color: item['color'] as Color),
                  const SizedBox(height: 6),
                  _loadingStats
                      ? Container(width: 24, height: 14,
                          decoration: BoxDecoration(
                            color: (item['color'] as Color).withOpacity(0.2),
                            borderRadius: BorderRadius.circular(4)))
                      : Text(item['val'] as String,
                          style: GoogleFonts.dmSans(
                            fontSize: 15, fontWeight: FontWeight.w800,
                            color: item['color'] as Color)),
                  const SizedBox(height: 2),
                  Text(item['label'] as String,
                    style: GoogleFonts.dmSans(
                      fontSize: 9, fontWeight: FontWeight.w600,
                      color: (item['color'] as Color).withOpacity(0.7),
                      letterSpacing: 0.3)),
                ]),
              ),
            );
          }).toList(),
        ),
      ),
    );
  }

  // ── Grid ─────────────────────────────────────────────────────────────────────
  Widget _buildGrid() {
    return AnimatedBuilder(
      animation: _gridAnim,
      builder: (_, child) => Opacity(opacity: _gridAnim.value, child: child),
      child: GridView.count(
        crossAxisCount: 3,
        shrinkWrap: true,
        physics: const NeverScrollableScrollPhysics(),
        mainAxisSpacing: 12,
        crossAxisSpacing: 12,
        childAspectRatio: 0.95,
        children: _actions.asMap().entries.map((e) {
          final a = e.value;
          final grads = a['gradient'] as List<Color>;
          return GestureDetector(
            onTap: () => Navigator.push(context,
                MaterialPageRoute(builder: (_) => a['screen'] as Widget)),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(18),
                boxShadow: [
                  BoxShadow(color: grads[1].withOpacity(0.12),
                      blurRadius: 12, offset: const Offset(0, 4))
                ]),
              child: Stack(children: [
                // Subtle gradient top accent
                Positioned(top: 0, left: 0, right: 0,
                  child: Container(
                    height: 3,
                    decoration: BoxDecoration(
                      gradient: LinearGradient(colors: grads),
                      borderRadius: const BorderRadius.vertical(
                          top: Radius.circular(18))))),

                Center(child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const SizedBox(height: 8),
                    Container(
                      width: 44, height: 44,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(
                          colors: grads,
                          begin: Alignment.topLeft,
                          end: Alignment.bottomRight),
                        borderRadius: BorderRadius.circular(13),
                        boxShadow: [BoxShadow(
                          color: grads[1].withOpacity(0.35),
                          blurRadius: 10, offset: const Offset(0, 4))]),
                      child: Icon(a['icon'] as IconData,
                          size: 21, color: Colors.white)),
                    const SizedBox(height: 8),
                    Text(a['label'] as String,
                      style: GoogleFonts.dmSans(
                        fontSize: 11, fontWeight: FontWeight.w700,
                        color: const Color(0xFF0D1F1F))),
                    Text(a['sublabel'] as String,
                      style: GoogleFonts.dmSans(
                        fontSize: 9, color: Colors.black38,
                        letterSpacing: 0.3)),
                  ])),

                // Badge
                if (a['badge'] != null)
                  Positioned(top: 10, right: 10,
                    child: Container(
                      width: 20, height: 20,
                      decoration: BoxDecoration(
                        gradient: LinearGradient(colors: grads),
                        shape: BoxShape.circle,
                        boxShadow: [BoxShadow(
                          color: grads[0].withOpacity(0.5),
                          blurRadius: 6)]),
                      child: Center(child: Text(a['badge'] as String,
                        style: GoogleFonts.dmSans(
                          fontSize: 9, fontWeight: FontWeight.w800,
                          color: Colors.white))))),
              ]),
            ),
          );
        }).toList(),
      ),
    );
  }

  // ── Activity ──────────────────────────────────────────────────────────────────
  Widget _buildActivity() {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(color: Colors.black.withOpacity(0.05),
              blurRadius: 16, offset: const Offset(0, 4))
        ]),
      child: Column(
        children: _activity.asMap().entries.map((e) {
          final a = e.value;
          final isLast = e.key == _activity.length - 1;
          return Container(
            padding: const EdgeInsets.fromLTRB(16, 14, 16, 14),
            decoration: BoxDecoration(
              border: isLast ? null : Border(
                bottom: BorderSide(color: Colors.black.withOpacity(0.05)))),
            child: Row(children: [
              Container(
                width: 40, height: 40,
                decoration: BoxDecoration(
                  color: (a['color'] as Color).withOpacity(0.08),
                  borderRadius: BorderRadius.circular(12)),
                child: Icon(a['icon'] as IconData,
                    size: 18, color: a['color'] as Color)),
              const SizedBox(width: 12),
              Expanded(child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(a['title'] as String,
                    style: GoogleFonts.dmSans(
                      fontSize: 12, fontWeight: FontWeight.w600,
                      color: const Color(0xFF0D1F1F))),
                  const SizedBox(height: 2),
                  Text(a['sub'] as String,
                    style: GoogleFonts.dmSans(
                      fontSize: 10, color: Colors.black38)),
                ])),
              Container(
                width: 6, height: 6,
                decoration: BoxDecoration(
                  color: (a['dot'] as Color).withOpacity(0.6),
                  shape: BoxShape.circle)),
            ]),
          );
        }).toList(),
      ),
    );
  }
}
