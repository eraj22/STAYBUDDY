import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../api.dart';
import '../../theme.dart';
import '../../widgets/parent/parent_portal.dart';
import '../../widgets/video_background.dart';

import 'parent_child_details_screen.dart';
import 'parent_fees_screen.dart';
import 'parent_attendance_screen.dart';
import 'parent_location_screen.dart';
import 'parent_transport_screen.dart';
import 'parent_voice_command_screen.dart';
import '../chatbot_screen.dart';
import 'parent_admin_chat_screen.dart';
import 'parent_child_chat_screen.dart';
import 'parent_announcements_screen.dart';
import 'parent_emergency_screen.dart';
import 'parent_notifications_screen.dart';
import 'parent_profile_screen.dart';

class ParentDashboardScreen extends StatefulWidget {
  const ParentDashboardScreen({super.key});

  @override
  State<ParentDashboardScreen> createState() => _ParentDashboardScreenState();
}

class _ParentDashboardScreenState extends State<ParentDashboardScreen> {
  final GlobalKey<NavigatorState> _parentNavigatorKey =
      GlobalKey<NavigatorState>();

  static const Color background = AppTheme.tealDeep;
  static const Color darkTeal = AppTheme.tealDark;
  static const Color teal = AppTheme.teal;
  static const Color mint = AppTheme.tealMint;
  static const Color green = Color(0xFF35EE96);

  Map<String, dynamic> _child = const {};
  Map<String, dynamic> _booking = const {};
  String _parentName = 'Parent';

  @override
  void initState() {
    super.initState();
    _loadLinkedChild();
  }

  Future<void> _loadLinkedChild() async {
    try {
      final parent = await AuthStore.getUser();
      final children = await Api().getParentChildren();
      if (children.isEmpty) {
        if (mounted) {
          setState(() {
            _parentName = parent?['name']?.toString() ?? 'Parent';
          });
        }
        return;
      }

      final firstChild = Map<String, dynamic>.from(children.first as Map);
      final studentId = int.tryParse(firstChild['id']?.toString() ?? '');
      Map<String, dynamic> child = firstChild;
      Map<String, dynamic> booking = const {};
      if (studentId != null) {
        final overview = await Api().getParentChildOverview(studentId);
        final student = overview['student'];
        if (student is Map) child = Map<String, dynamic>.from(student);
        final bookings = overview['bookings'];
        if (bookings is List && bookings.isNotEmpty) {
          final preferred = bookings.cast<dynamic>().firstWhere(
            (item) => item is Map && item['status'] == 'confirmed',
            orElse: () => bookings.first,
          );
          if (preferred is Map) booking = Map<String, dynamic>.from(preferred);
        }
      }

      if (mounted) {
        setState(() {
          _parentName = parent?['name']?.toString() ?? 'Parent';
          _child = child;
          _booking = booking;
        });
      }
    } catch (_) {
      // The screen stays usable; it simply shows no linked child until the
      // backend is available again.
    }
  }

  String get _childName => _child['name']?.toString() ?? 'No linked student';
  String get _programme =>
      _child['department']?.toString().trim().isNotEmpty == true
      ? _child['department'].toString()
      : 'Student profile';
  String get _university =>
      _child['university']?.toString().trim().isNotEmpty == true
      ? _child['university'].toString()
      : _booking['hostel_name']?.toString() ?? 'Hostel details unavailable';
  String get _roomLabel => _booking['room_number'] == null
      ? 'Room not assigned'
      : 'Room ${_booking['room_number']}';
  String get _bedLabel => _booking['bed_number'] == null
      ? 'Bed not assigned'
      : 'Bed ${_booking['bed_number']}';
  bool get _isInsideHostel =>
      _booking['checked_in_at'] != null && _booking['checked_out_at'] == null;

  void _openScreen(Widget screen) {
    _parentNavigatorKey.currentState?.push(
      MaterialPageRoute(builder: (_) => screen),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ParentPortal(
      navigatorKey: _parentNavigatorKey,
      builder: (_) => _buildDashboard(),
    );
  }

  Widget _buildDashboard() {
    return Scaffold(
      backgroundColor: Colors.transparent,
      body: VideoBackground(
        assetPath: 'assets/images/background.mp4',
        overlayOpacity: .18,
        child: Stack(
          children: [
            SafeArea(
              child: LayoutBuilder(
                builder: (context, constraints) {
                  return SingleChildScrollView(
                    physics: const BouncingScrollPhysics(),
                    padding: const EdgeInsets.fromLTRB(18, 10, 18, 110),
                    child: Center(
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(maxWidth: 600),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            _buildTopBar(),

                            const SizedBox(height: 22),

                            _buildChildCard(),

                            const SizedBox(height: 24),

                            _sectionTitle(
                              'Parent Services',
                              'Everything you need to stay connected',
                            ),

                            const SizedBox(height: 14),

                            _buildServicesGrid(),

                            const SizedBox(height: 24),

                            _sectionTitle(
                              'Important',
                              'Recent updates for your child',
                            ),

                            const SizedBox(height: 14),

                            _buildFeeReminder(),

                            const SizedBox(height: 12),

                            _buildAttendanceUpdate(),

                            const SizedBox(height: 12),

                            _buildAnnouncement(),
                          ],
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
            _buildBottomNavigation(),
          ],
        ),
      ),
    );
  }

  // =========================================================
  // TOP BAR
  // =========================================================

  Widget _buildTopBar() {
    return Row(
      children: [
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'StayBuddy',
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.playfairDisplay(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.w700,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                _parentName,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.dmSans(
                  color: mint,
                  fontSize: 11.5,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(width: 12),

        Container(
          padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 8),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(.24),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white.withOpacity(.16)),
          ),
          child: Text(
            'Parent Dashboard',
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 9.5,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ],
    );
  }

  Widget _circleButton({required IconData icon, required VoidCallback onTap}) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(50),
        child: Container(
          width: 44,
          height: 44,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: Colors.black.withOpacity(.26),
            border: Border.all(color: Colors.white.withOpacity(.16)),
          ),
          child: Icon(icon, color: Colors.white, size: 22),
        ),
      ),
    );
  }

  // =========================================================
  // PARENT MENU
  // =========================================================

  void _showParentMenu() {
    final portalContext =
        _parentNavigatorKey.currentState?.overlay?.context ?? context;
    showModalBottomSheet(
      context: portalContext,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (_) {
        return SafeArea(
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.fromLTRB(18, 18, 18, 22),
            decoration: BoxDecoration(
              color: const Color(0xFF003333),
              borderRadius: BorderRadius.circular(26),
              border: Border.all(color: mint.withOpacity(.12)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Container(
                  width: 42,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.white24,
                    borderRadius: BorderRadius.circular(20),
                  ),
                ),

                const SizedBox(height: 18),

                Text(
                  'Parent Menu',
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                  ),
                ),

                const SizedBox(height: 16),

                _menuItem(
                  icon: Icons.person_outline_rounded,
                  title: 'Profile & Settings',
                  onTap: () {
                    Navigator.pop(portalContext);

                    _openScreen(const ParentProfileScreen());
                  },
                ),

                _menuItem(
                  icon: Icons.notifications_none_rounded,
                  title: 'Notifications',
                  onTap: () {
                    Navigator.pop(portalContext);

                    _openScreen(const ParentNotificationsScreen());
                  },
                ),

                _menuItem(
                  icon: Icons.support_agent_rounded,
                  title: 'Contact Admin',
                  onTap: () {
                    Navigator.pop(portalContext);

                    _openScreen(const ParentAdminChatScreen());
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _menuItem({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 12),
          child: Row(
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: teal.withOpacity(.12),
                  borderRadius: BorderRadius.circular(13),
                ),
                child: Icon(icon, color: mint, size: 20),
              ),

              const SizedBox(width: 12),

              Expanded(
                child: Text(
                  title,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),

              const Icon(Icons.chevron_right_rounded, color: Colors.white30),
            ],
          ),
        ),
      ),
    );
  }

  // =========================================================
  // CHILD CARD
  // =========================================================

  Widget _buildChildCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF00645E).withOpacity(.95),
            const Color(0xFF003C3C).withOpacity(.95),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: mint.withOpacity(.22)),
        boxShadow: [
          BoxShadow(
            color: teal.withOpacity(.12),
            blurRadius: 26,
            spreadRadius: 2,
          ),
        ],
      ),
      child: Column(
        children: [
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Container(
                width: 74,
                height: 74,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: Colors.white.withOpacity(.10),
                  border: Border.all(color: mint, width: 1.5),
                ),
                child: const Icon(
                  Icons.person_rounded,
                  color: Colors.white,
                  size: 45,
                ),
              ),

              const SizedBox(width: 14),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _childName,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 21,
                        fontWeight: FontWeight.w800,
                      ),
                    ),

                    const SizedBox(height: 4),

                    Text(
                      _programme,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white70,
                        fontSize: 12,
                      ),
                    ),

                    const SizedBox(height: 3),

                    Text(
                      _university,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: mint,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                      ),
                    ),

                    const SizedBox(height: 8),

                    Wrap(
                      spacing: 8,
                      runSpacing: 8,
                      children: [
                        _smallInfoChip(Icons.meeting_room_outlined, _roomLabel),
                        _smallInfoChip(Icons.bed_outlined, _bedLabel),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 18),

          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
            decoration: BoxDecoration(
              color: Colors.black.withOpacity(.12),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Row(
              children: [
                Container(
                  width: 9,
                  height: 9,
                  decoration: const BoxDecoration(
                    color: green,
                    shape: BoxShape.circle,
                  ),
                ),

                const SizedBox(width: 8),

                Expanded(
                  child: Text(
                    _isInsideHostel
                        ? 'Currently inside hostel'
                        : 'Not currently checked in',
                    overflow: TextOverflow.ellipsis,
                    style: GoogleFonts.dmSans(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 12,
                    ),
                  ),
                ),

                Text(
                  _isInsideHostel ? 'IN' : 'OUT',
                  style: GoogleFonts.dmSans(
                    color: green,
                    fontWeight: FontWeight.w800,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _smallInfoChip(IconData icon, String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 7),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(.07),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: mint, size: 15),
          const SizedBox(width: 5),
          Text(
            text,
            style: GoogleFonts.dmSans(
              color: Colors.white70,
              fontSize: 10.5,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  // =========================================================
  // SECTION TITLE
  // =========================================================

  Widget _sectionTitle(String title, String subtitle) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: GoogleFonts.dmSans(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w800,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          subtitle,
          style: GoogleFonts.dmSans(color: Colors.white38, fontSize: 11),
        ),
      ],
    );
  }

  // =========================================================
  // SERVICES GRID
  // =========================================================

  Widget _buildServicesGrid() {
    final services = [
      _ParentService(
        title: 'Child Details',
        icon: Icons.badge_outlined,
        onTap: () {
          _openScreen(const ParentChildDetailsScreen());
        },
      ),

      _ParentService(
        title: 'Fees',
        icon: Icons.account_balance_wallet_outlined,
        onTap: () {
          _openScreen(const ParentFeesScreen());
        },
      ),

      _ParentService(
        title: 'Attendance',
        icon: Icons.fact_check_outlined,
        onTap: () {
          _openScreen(const ParentAttendanceScreen());
        },
      ),

      _ParentService(
        title: 'Live Location',
        icon: Icons.location_on_outlined,
        onTap: () {
          _openScreen(const ParentLocationScreen());
        },
      ),

      _ParentService(
        title: 'Transport',
        icon: Icons.directions_bus_outlined,
        onTap: () {
          _openScreen(const ParentTransportScreen());
        },
      ),

      _ParentService(
        title: 'Voice Command',
        icon: Icons.mic_none_rounded,
        onTap: () {
          _openScreen(const ParentVoiceCommandScreen());
        },
      ),

      _ParentService(
        title: 'AI Assistant',
        icon: Icons.smart_toy_outlined,
        onTap: () {
          _openScreen(const ChatbotScreen());
        },
      ),

      _ParentService(
        title: 'Chat Admin',
        icon: Icons.support_agent_rounded,
        onTap: () {
          _openScreen(const ParentAdminChatScreen());
        },
      ),

      _ParentService(
        title: 'Chat Child',
        icon: Icons.chat_bubble_outline_rounded,
        onTap: () {
          _openScreen(const ParentChildChatScreen());
        },
      ),

      _ParentService(
        title: 'Announcements',
        icon: Icons.campaign_outlined,
        onTap: () {
          _openScreen(const ParentAnnouncementsScreen());
        },
      ),

      _ParentService(
        title: 'Emergency',
        icon: Icons.emergency_outlined,
        danger: true,
        onTap: () {
          _openScreen(const ParentEmergencyScreen());
        },
      ),
    ];

    return LayoutBuilder(
      builder: (context, constraints) {
        final bool narrow = constraints.maxWidth < 370;

        return GridView.builder(
          shrinkWrap: true,
          physics: const NeverScrollableScrollPhysics(),
          itemCount: services.length,
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: 2,
            crossAxisSpacing: 12,
            mainAxisSpacing: 12,
            childAspectRatio: narrow ? 1.55 : 1.72,
          ),
          itemBuilder: (_, index) {
            return _serviceCard(services[index]);
          },
        );
      },
    );
  }

  Widget _serviceCard(_ParentService service) {
    final Color iconColor = service.danger ? Colors.redAccent : mint;

    final Color cardBorder = service.danger
        ? Colors.redAccent.withOpacity(.30)
        : mint.withOpacity(.13);

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: service.onTap,
        borderRadius: BorderRadius.circular(22),
        child: Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 13),
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: service.danger
                  ? [
                      const Color(0xFF40282B).withOpacity(.94),
                      const Color(0xFF073F41).withOpacity(.96),
                    ]
                  : [
                      const Color(0xFF07585A).withOpacity(.94),
                      const Color(0xFF033D40).withOpacity(.96),
                    ],
            ),
            borderRadius: BorderRadius.circular(22),
            border: Border.all(color: cardBorder),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(.24),
                blurRadius: 16,
                offset: const Offset(0, 7),
              ),
            ],
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                width: 42,
                height: 42,
                decoration: BoxDecoration(
                  color: service.danger
                      ? Colors.redAccent.withOpacity(.12)
                      : mint.withOpacity(.10),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(service.icon, color: iconColor, size: 23),
              ),

              const SizedBox(height: 8),

              Text(
                service.title,
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                textAlign: TextAlign.center,
                style: GoogleFonts.dmSans(
                  color: service.danger
                      ? const Color(0xFFFF9A9A)
                      : Colors.white,
                  fontSize: 11.5,
                  height: 1.25,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // =========================================================
  // IMPORTANT CARDS
  // =========================================================

  Widget _buildFeeReminder() {
    return _infoCard(
      icon: Icons.receipt_long_rounded,
      iconColor: mint,
      title: 'Fee Due Reminder',
      description: 'August hostel fee is due in 5 days.',
      trailing: 'View',
      onTap: () {
        _openScreen(const ParentFeesScreen());
      },
    );
  }

  Widget _buildAttendanceUpdate() {
    return _infoCard(
      icon: Icons.login_rounded,
      iconColor: green,
      title: '$_childName hostel update',
      description: 'Today at 7:45 AM',
      trailing: 'IN',
      trailingColor: green,
      onTap: () {
        _openScreen(const ParentAttendanceScreen());
      },
    );
  }

  Widget _buildAnnouncement() {
    return _infoCard(
      icon: Icons.campaign_outlined,
      iconColor: const Color(0xFFFFD166),
      title: 'Hostel Announcement',
      description: 'Mess timing has been updated.',
      trailing: 'Read',
      onTap: () {
        _openScreen(const ParentAnnouncementsScreen());
      },
    );
  }

  Widget _infoCard({
    required IconData icon,
    required Color iconColor,
    required String title,
    required String description,
    required String trailing,
    required VoidCallback onTap,
    Color trailingColor = mint,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(20),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.all(15),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(.32),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white.withOpacity(.08)),
          ),
          child: Row(
            children: [
              Container(
                width: 45,
                height: 45,
                decoration: BoxDecoration(
                  color: iconColor.withOpacity(.12),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Icon(icon, color: iconColor, size: 22),
              ),

              const SizedBox(width: 12),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 13,
                        fontWeight: FontWeight.w700,
                      ),
                    ),

                    const SizedBox(height: 3),

                    Text(
                      description,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.45),
                        fontSize: 10.5,
                        height: 1.3,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(width: 8),

              Text(
                trailing,
                style: GoogleFonts.dmSans(
                  color: trailingColor,
                  fontSize: 11,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // =========================================================
  // BOTTOM NAVIGATION
  // =========================================================

  Widget _buildBottomNavigation() {
    return Positioned(
      left: 16,
      right: 16,
      bottom: 14,
      child: SafeArea(
        top: false,
        child: Container(
          height: 68,
          padding: const EdgeInsets.symmetric(horizontal: 18),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(.72),
            borderRadius: BorderRadius.circular(24),
            border: Border.all(color: Colors.white.withOpacity(.18)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(.35),
                blurRadius: 18,
                offset: const Offset(0, 8),
              ),
            ],
          ),
          child: Row(
            children: [
              _navItem(
                icon: Icons.home_rounded,
                label: 'Home',
                active: true,
                onTap: () {},
              ),

              _navItem(
                icon: Icons.notifications_none_rounded,
                label: 'Alerts',
                onTap: () {
                  _openScreen(const ParentNotificationsScreen());
                },
              ),

              _navItem(
                icon: Icons.person_outline_rounded,
                label: 'Profile',
                onTap: () {
                  _openScreen(const ParentProfileScreen());
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _navItem({
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    bool active = false,
  }) {
    return Expanded(
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(18),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(icon, size: 21, color: active ? mint : Colors.white30),

            const SizedBox(height: 4),

            Text(
              label,
              maxLines: 1,
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: active ? mint : Colors.white30,
                fontSize: 9,
                fontWeight: active ? FontWeight.w700 : FontWeight.w500,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ParentService {
  final String title;
  final IconData icon;
  final VoidCallback onTap;
  final bool danger;

  const _ParentService({
    required this.title,
    required this.icon,
    required this.onTap,
    this.danger = false,
  });
}
