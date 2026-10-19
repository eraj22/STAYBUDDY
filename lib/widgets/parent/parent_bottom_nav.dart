import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../../theme.dart';

class ParentBottomNav extends StatelessWidget {
  final int currentIndex;

  final VoidCallback onHomeTap;
  final VoidCallback onAlertsTap;
  final VoidCallback onProfileTap;

  const ParentBottomNav({
    super.key,
    required this.currentIndex,
    required this.onHomeTap,
    required this.onAlertsTap,
    required this.onProfileTap,
  });

  @override
  Widget build(BuildContext context) {
    return SafeArea(
      top: false,
      minimum: const EdgeInsets.fromLTRB(
        16,
        0,
        16,
        14,
      ),
      child: Container(
        height: 68,
        padding: const EdgeInsets.symmetric(
          horizontal: 16,
        ),
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.72),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: Colors.white.withOpacity(0.18),
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.35),
              blurRadius: 18,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Row(
          children: [
            _ParentNavItem(
              icon: Icons.home_rounded,
              label: 'Home',
              active: currentIndex == 0,
              onTap: onHomeTap,
            ),

            _ParentNavItem(
              icon: Icons.notifications_none_rounded,
              label: 'Alerts',
              active: currentIndex == 1,
              onTap: onAlertsTap,
            ),

            _ParentNavItem(
              icon: Icons.person_outline_rounded,
              label: 'Profile',
              active: currentIndex == 2,
              onTap: onProfileTap,
            ),
          ],
        ),
      ),
    );
  }
}

class _ParentNavItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final bool active;
  final VoidCallback onTap;

  const _ParentNavItem({
    required this.icon,
    required this.label,
    required this.active,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(18),
          child: SizedBox(
            height: 62,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                AnimatedContainer(
                  duration: const Duration(
                    milliseconds: 180,
                  ),
                  padding: EdgeInsets.symmetric(
                    horizontal: active ? 15 : 10,
                    vertical: 5,
                  ),
                  decoration: BoxDecoration(
                    color: active
                        ? AppTheme.teal.withOpacity(0.24)
                        : Colors.transparent,
                    borderRadius:
                        BorderRadius.circular(18),
                  ),
                  child: Icon(
                    icon,
                    size: 21,
                    color: active
                        ? AppTheme.tealMint
                        : Colors.white30,
                  ),
                ),

                const SizedBox(height: 2),

                Text(
                  label,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: active
                        ? AppTheme.tealMint
                        : Colors.white30,
                    fontSize: 9,
                    fontWeight: active
                        ? FontWeight.w700
                        : FontWeight.w500,
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
