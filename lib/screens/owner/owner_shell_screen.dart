import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../theme.dart';
import '../../widgets/responsive_shell.dart';
import '../../widgets/video_background.dart';

import 'dashboard/owner_dashboard_screen.dart';
import 'students/owner_students_screen.dart';
import 'rooms/owner_rooms_screen.dart';
import 'reports/owner_reports_screen.dart';
import 'settings/owner_settings_screen.dart';

class OwnerShellScreen extends StatefulWidget {
  final String ownerName;
  final String ownerEmail;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;

  const OwnerShellScreen({
    super.key,
    required this.ownerName,
    required this.ownerEmail,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
  });

  @override
  State<OwnerShellScreen> createState() => _OwnerShellScreenState();
}

class _OwnerShellScreenState extends State<OwnerShellScreen> {
  int _currentIndex = 0;

  late final List<Widget> _pages;

  @override
  void initState() {
    super.initState();

    _pages = [
      OwnerDashboardScreen(
        ownerName: widget.ownerName,
        ownerEmail: widget.ownerEmail,
        hostelName: widget.hostelName,
        hostelType: widget.hostelType,
        city: widget.city,
        address: widget.address,
      ),

      OwnerStudentsScreen(
        hostelName: widget.hostelName,
        onBackToDashboard: _showDashboard,
      ),

      OwnerRoomsScreen(
        hostelName: widget.hostelName,
        onBackToDashboard: _showDashboard,
      ),

      OwnerReportsScreen(
        hostelName: widget.hostelName,
        onBackToDashboard: _showDashboard,
      ),

      OwnerSettingsScreen(
        hostelName: widget.hostelName,
        onBackToDashboard: _showDashboard,
      ),
    ];
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: PopScope<void>(
        canPop: _currentIndex == 0,
        onPopInvokedWithResult: (didPop, _) {
          if (!didPop && _currentIndex != 0) {
            _showDashboard();
          }
        },
        child: Scaffold(
          backgroundColor: Colors.transparent,
          extendBody: true,
          body: VideoBackground(
            assetPath: 'assets/images/background.mp4',
            overlayOpacity: 0.14,
            tintColor: AppTheme.tealDeep,
            tintOpacity: 0.58,
            cardMargin: const EdgeInsets.all(14),
            cardRadius: 32,
            child: IndexedStack(index: _currentIndex, children: _pages),
          ),
          bottomNavigationBar: _buildBottomNavigation(),
        ),
      ),
    );
  }

  void _showDashboard() {
    if (!mounted || _currentIndex == 0) return;
    setState(() => _currentIndex = 0);
  }

  Widget _buildBottomNavigation() {
    final bottomPadding = MediaQuery.of(context).padding.bottom;

    return Container(
      margin: EdgeInsets.fromLTRB(14, 0, 14, bottomPadding > 0 ? 8 : 14),
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
      decoration: BoxDecoration(
        color: AppTheme.tealDeep.withValues(alpha: 0.92),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(
          color: Colors.white.withValues(alpha: 0.16),
          width: 0.7,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.28),
            blurRadius: 28,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: Row(
        children: [
          _navItem(
            index: 0,
            icon: Icons.home_outlined,
            selectedIcon: Icons.home_rounded,
            label: 'Home',
          ),
          _navItem(
            index: 1,
            icon: Icons.groups_outlined,
            selectedIcon: Icons.groups_rounded,
            label: 'Students',
          ),
          _navItem(
            index: 2,
            icon: Icons.bed_outlined,
            selectedIcon: Icons.bed_rounded,
            label: 'Rooms',
          ),
          _navItem(
            index: 3,
            icon: Icons.insights_outlined,
            selectedIcon: Icons.insights_rounded,
            label: 'Reports',
          ),
          _navItem(
            index: 4,
            icon: Icons.settings_outlined,
            selectedIcon: Icons.settings_rounded,
            label: 'Settings',
          ),
        ],
      ),
    );
  }

  Widget _navItem({
    required int index,
    required IconData icon,
    required IconData selectedIcon,
    required String label,
  }) {
    final selected = _currentIndex == index;

    return Expanded(
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () {
            if (_currentIndex == index) return;

            setState(() {
              _currentIndex = index;
            });
          },
          borderRadius: BorderRadius.circular(18),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 220),
            curve: Curves.easeOutCubic,
            padding: const EdgeInsets.symmetric(horizontal: 5, vertical: 9),
            decoration: BoxDecoration(
              color: selected
                  ? Colors.white.withValues(alpha: 0.10)
                  : Colors.transparent,
              borderRadius: BorderRadius.circular(18),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                AnimatedContainer(
                  duration: const Duration(milliseconds: 220),
                  width: selected ? 38 : 34,
                  height: 30,
                  decoration: BoxDecoration(
                    color: selected ? AppTheme.teal : Colors.transparent,
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    selected ? selectedIcon : icon,
                    size: selected ? 19 : 20,
                    color: selected
                        ? Colors.white
                        : Colors.white.withValues(alpha: 0.68),
                  ),
                ),

                const SizedBox(height: 4),

                Text(
                  label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 9,
                    fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
                    color: selected
                        ? AppTheme.teal
                        : Colors.white.withValues(alpha: 0.68),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
