import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import 'owner_manage_wardens_screen.dart';
import 'owner_hostel_profile_screen.dart';

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
  // Demo stats
  final _stats = {
    'students': 62,
    'wardens': 2,
    'complaints': 4,
    'revenue': '186,000',
    'occupied': 44,
    'total': 62,
  };

  final _recentActivity = [
    {'icon': Icons.person_add_rounded,    'color': Color(0xFF2D6A11),  'title': 'New booking request',      'sub': 'Sara Khan · Room 102 · 2 hrs ago'},
    {'icon': Icons.report_rounded,        'color': Color(0xFFB85C00),  'title': 'New complaint filed',      'sub': 'WiFi issue · Amna Raza · 5 hrs ago'},
    {'icon': Icons.payments_rounded,      'color': Color(0xFF185FA5),  'title': 'Fee payment received',     'sub': 'PKR 15,000 · Room 201 · Yesterday'},
    {'icon': Icons.campaign_rounded,      'color': AppTheme.teal,      'title': 'Warden posted announcement','sub': 'Mess timing update · Yesterday'},
    {'icon': Icons.bed_rounded,           'color': Color(0xFF7B3FC4),  'title': 'Bed vacancy alert',        'sub': 'Room 301 · 1 bed available · 2 days ago'},
  ];

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        body: CustomScrollView(
          slivers: [
            SliverToBoxAdapter(child: _buildHeader()),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 18, 16, 0),
              sliver: SliverToBoxAdapter(child: _buildStats()),
            ),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 22, 16, 8),
              sliver: SliverToBoxAdapter(child: _sectionTitle('Quick Actions')),
            ),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 0),
              sliver: SliverToBoxAdapter(child: _buildActions()),
            ),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 22, 16, 8),
              sliver: SliverToBoxAdapter(child: _sectionTitle('Recent Activity')),
            ),
            SliverPadding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 32),
              sliver: SliverToBoxAdapter(child: _buildActivity()),
            ),
          ],
        ),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────
  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF0A3D3F), AppTheme.teal],
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 14, 20, 22),
          child: Column(children: [
            Row(children: [
              Container(
                width: 48, height: 48,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.15),
                  shape: BoxShape.circle,
                  border: Border.all(color: Colors.white.withOpacity(0.3))),
                child: const Icon(Icons.person_rounded,
                    color: Colors.white, size: 24)),
              const SizedBox(width: 12),
              Expanded(child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text('Welcome back,',
                    style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.65), fontSize: 12)),
                Text(widget.ownerName,
                    style: GoogleFonts.playfairDisplay(
                        fontSize: 20, fontWeight: FontWeight.w700,
                        color: Colors.white)),
              ])),
              // Notification bell
              Container(
                width: 38, height: 38,
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.15),
                  shape: BoxShape.circle),
                child: Stack(children: [
                  const Center(child: Icon(Icons.notifications_rounded,
                      color: Colors.white, size: 20)),
                  Positioned(top: 8, right: 8,
                    child: Container(width: 8, height: 8,
                      decoration: const BoxDecoration(
                          color: Colors.red, shape: BoxShape.circle))),
                ]),
              ),
            ]),
            const SizedBox(height: 14),
            // Hostel info card
            GestureDetector(
              onTap: () => Navigator.push(context, MaterialPageRoute(
                builder: (_) => OwnerHostelProfileScreen(
                  ownerName: widget.ownerName,
                  ownerEmail: widget.ownerEmail,
                  hostelName: widget.hostelName,
                  hostelType: widget.hostelType,
                  city: widget.city,
                  address: widget.address,
                ))),
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.white.withOpacity(0.2))),
                child: Row(children: [
                  Container(
                    width: 40, height: 40,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(10)),
                    child: const Icon(Icons.apartment_rounded,
                        color: Colors.white, size: 20)),
                  const SizedBox(width: 12),
                  Expanded(child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(widget.hostelName,
                        style: GoogleFonts.dmSans(
                            fontSize: 14, fontWeight: FontWeight.w700,
                            color: Colors.white)),
                    Text('${widget.city} · ${widget.hostelType}',
                        style: GoogleFonts.dmSans(
                            fontSize: 11, color: Colors.white60)),
                  ])),
                  Row(children: [
                    Container(width: 7, height: 7,
                      decoration: const BoxDecoration(
                          color: AppTheme.tealMint, shape: BoxShape.circle)),
                    const SizedBox(width: 5),
                    Text('Live', style: GoogleFonts.dmSans(
                        color: AppTheme.tealMint, fontSize: 11,
                        fontWeight: FontWeight.w600)),
                  ]),
                ]),
              ),
            ),
          ]),
        ),
      ),
    );
  }

  // ── Stats ─────────────────────────────────────────────────────────────────
  Widget _buildStats() {
    final items = [
      {'val': '${_stats['students']}',  'label': 'Students',   'icon': Icons.school_rounded,        'color': AppTheme.teal},
      {'val': '${_stats['wardens']}',   'label': 'Wardens',    'icon': Icons.badge_rounded,          'color': Color(0xFF185FA5)},
      {'val': '${_stats['complaints']}','label': 'Complaints', 'icon': Icons.report_rounded,         'color': Color(0xFFB85C00)},
      {'val': 'PKR\n${_stats['revenue']}','label':'Revenue',   'icon': Icons.payments_rounded,       'color': Color(0xFF2D6A11)},
    ];
    return Row(
      children: items.asMap().entries.map((e) => Expanded(
        child: Container(
          margin: EdgeInsets.only(right: e.key < items.length - 1 ? 10 : 0),
          padding: const EdgeInsets.symmetric(vertical: 14),
          decoration: BoxDecoration(
            color: AppTheme.card,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(color: AppTheme.border, width: 0.5)),
          child: Column(children: [
            Icon(e.value['icon'] as IconData,
                size: 18, color: e.value['color'] as Color),
            const SizedBox(height: 6),
            Text(e.value['val'] as String,
                textAlign: TextAlign.center,
                style: GoogleFonts.dmSans(
                    fontSize: e.key == 3 ? 11 : 16,
                    fontWeight: FontWeight.w700,
                    color: e.value['color'] as Color, height: 1.2)),
            Text(e.value['label'] as String,
                style: GoogleFonts.dmSans(
                    fontSize: 10, color: AppTheme.textMuted)),
          ]),
        ),
      )).toList(),
    );
  }

  // ── Actions ───────────────────────────────────────────────────────────────
  Widget _buildActions() {
    final actions = [
      {
        'label': 'Hostel Profile',
        'icon': Icons.apartment_rounded,
        'color': AppTheme.teal,
        'bg': AppTheme.teal.withOpacity(0.08),
        'onTap': () => Navigator.push(context, MaterialPageRoute(
          builder: (_) => OwnerHostelProfileScreen(
            ownerName: widget.ownerName,
            ownerEmail: widget.ownerEmail,
            hostelName: widget.hostelName,
            hostelType: widget.hostelType,
            city: widget.city,
            address: widget.address,
          ))),
      },
      {
        'label': 'Manage\nWardens',
        'icon': Icons.badge_rounded,
        'color': Color(0xFF185FA5),
        'bg': Color(0xFFE3EEF9),
        'onTap': () => Navigator.push(context, MaterialPageRoute(
          builder: (_) => OwnerManageWardensScreen(
            ownerName: widget.ownerName,
            hostelName: widget.hostelName,
          ))),
      },
      {
        'label': 'Students',
        'icon': Icons.people_rounded,
        'color': Color(0xFF7B3FC4),
        'bg': Color(0xFFF0EAFC),
        'onTap': () {},
      },
      {
        'label': 'Complaints',
        'icon': Icons.report_rounded,
        'color': Color(0xFFB85C00),
        'bg': Color(0xFFFEF3DF),
        'onTap': () {},
      },
      {
        'label': 'Rooms &\nBeds',
        'icon': Icons.bed_rounded,
        'color': Color(0xFF2D6A11),
        'bg': AppTheme.greenLight,
        'onTap': () {},
      },
      {
        'label': 'Analytics',
        'icon': Icons.bar_chart_rounded,
        'color': AppTheme.teal,
        'bg': AppTheme.teal.withOpacity(0.08),
        'onTap': () {},
      },
      {
        'label': 'Fee\nManagement',
        'icon': Icons.payments_rounded,
        'color': Color(0xFF185FA5),
        'bg': Color(0xFFE3EEF9),
        'onTap': () {},
      },
      {
        'label': 'Settings',
        'icon': Icons.settings_rounded,
        'color': AppTheme.textMuted,
        'bg': AppTheme.bgSecondary,
        'onTap': () {},
      },
    ];

    return GridView.count(
      crossAxisCount: 4,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      mainAxisSpacing: 10,
      crossAxisSpacing: 10,
      childAspectRatio: 0.9,
      children: actions.map((a) => GestureDetector(
        onTap: a['onTap'] as VoidCallback,
        child: Container(
          decoration: BoxDecoration(
            color: a['bg'] as Color,
            borderRadius: BorderRadius.circular(14),
            border: Border.all(
                color: (a['color'] as Color).withOpacity(0.2),
                width: 0.5)),
          child: Column(
              mainAxisAlignment: MainAxisAlignment.center, children: [
            Container(
              width: 38, height: 38,
              decoration: BoxDecoration(
                color: (a['color'] as Color).withOpacity(0.12),
                shape: BoxShape.circle),
              child: Icon(a['icon'] as IconData,
                  size: 18, color: a['color'] as Color)),
            const SizedBox(height: 6),
            Text(a['label'] as String,
                textAlign: TextAlign.center,
                style: GoogleFonts.dmSans(
                    fontSize: 10, fontWeight: FontWeight.w600,
                    color: a['color'] as Color, height: 1.2)),
          ]),
        ),
      )).toList(),
    );
  }

  // ── Activity ──────────────────────────────────────────────────────────────
  Widget _buildActivity() {
    return Column(
      children: _recentActivity.map((a) => Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.all(13),
        decoration: BoxDecoration(
          color: AppTheme.card,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppTheme.border, width: 0.5)),
        child: Row(children: [
          Container(
            width: 38, height: 38,
            decoration: BoxDecoration(
              color: (a['color'] as Color).withOpacity(0.1),
              shape: BoxShape.circle),
            child: Icon(a['icon'] as IconData,
                size: 18, color: a['color'] as Color)),
          const SizedBox(width: 12),
          Expanded(child: Column(
              crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(a['title'] as String,
                style: GoogleFonts.dmSans(
                    fontSize: 13, fontWeight: FontWeight.w600,
                    color: AppTheme.textPrimary)),
            Text(a['sub'] as String,
                style: GoogleFonts.dmSans(
                    fontSize: 11, color: AppTheme.textMuted)),
          ])),
        ]),
      )).toList(),
    );
  }

  Widget _sectionTitle(String text) => Text(text,
      style: GoogleFonts.dmSans(
          fontSize: 15, fontWeight: FontWeight.w700,
          color: AppTheme.textPrimary));
}