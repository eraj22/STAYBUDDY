import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../routes.dart';
import '../theme.dart';
import '../widgets/video_scaffold.dart';
import '../widgets/glass_card.dart';
import '../widgets/three_d_container.dart';
import '../widgets/app_button.dart';
import '../theme/app_sizes.dart';
import 'warden_login_screen.dart';

class RoleScreen extends StatefulWidget {
  const RoleScreen({super.key});
  @override
  State<RoleScreen> createState() => _RoleScreenState();
}

class _RoleScreenState extends State<RoleScreen>
    with SingleTickerProviderStateMixin {
  String? selectedRole;
  late final AnimationController _controller;
  late final Animation<double> _fade;
  late final Animation<Offset> _slide;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );
    _fade = CurvedAnimation(parent: _controller, curve: Curves.easeOut);
    _slide = Tween<Offset>(begin: const Offset(0, 0.08), end: Offset.zero)
        .animate(CurvedAnimation(parent: _controller, curve: Curves.easeOutCubic));
    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _proceed() {
    if (selectedRole == null) return;
    switch (selectedRole) {
      case 'Owner':
        Navigator.pushReplacementNamed(context, Routes.ownerLogin);
        break;
      case 'Student':
        Navigator.pushReplacementNamed(context, Routes.login);
        break;
      case 'Warden':
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (_) => const WardenLoginScreen()),
        );
        break;
      default:
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('$selectedRole flow coming soon!'),
            backgroundColor: AppTheme.teal,
            behavior: SnackBarBehavior.floating,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
        );
    }
  }

  @override
  Widget build(BuildContext context) {
    return VideoScaffold(
      child: FadeTransition(
        opacity: _fade,
        child: SlideTransition(
          position: _slide,
          child: SafeArea(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 6),
                  Text(
                    'Welcome to StayBuddy',
                    style: GoogleFonts.dmSans(
                      color: Colors.white.withOpacity(0.75),
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Who are you?',
                    style: GoogleFonts.playfairDisplay(
                      fontSize: 32,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Select your role to get a tailored experience.',
                    style: GoogleFonts.dmSans(
                      color: Colors.white.withOpacity(0.65),
                      fontSize: 13,
                    ),
                  ),
                  const SizedBox(height: 28),

                  Expanded(
                    child: ListView.separated(
                      physics: const BouncingScrollPhysics(),
                      itemCount: _roles.length,
                      separatorBuilder: (_, __) => const SizedBox(height: 14),
                      itemBuilder: (_, i) {
                        final r = _roles[i];
                        return ThreeDContainer(
                          borderRadius: 20,
                          maxTilt: 6,
                          child: _RoleCard(
                            role: r,
                            isSelected: selectedRole == r.label,
                            onTap: () => setState(() => selectedRole = r.label),
                          ),
                        );
                      },
                    ),
                  ),

                  const SizedBox(height: 12),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Expanded(
                        child: AnimatedSwitcher(
                          duration: const Duration(milliseconds: 200),
                          child: selectedRole != null
                              ? Text(
                                  'Continuing as $selectedRole',
                                  key: ValueKey(selectedRole),
                                  style: GoogleFonts.dmSans(
                                    color: Colors.white.withOpacity(0.75),
                                    fontSize: 13,
                                    fontWeight: FontWeight.w500,
                                  ),
                                )
                              : const SizedBox(),
                        ),
                      ),
                      AnimatedOpacity(
                        duration: const Duration(milliseconds: 250),
                        opacity: selectedRole != null ? 1.0 : 0.35,
                        child: AppCircleButton(
                          icon: Icons.arrow_forward_rounded,
                          size: AppSizes.fabSize,
                          useGradient: true,
                          onPressed: selectedRole != null ? _proceed : null,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _RoleData {
  final String label, description;
  final IconData icon;
  const _RoleData({required this.label, required this.description, required this.icon});
}

const _roles = [
  _RoleData(label: 'Student', description: 'Find and explore verified hostels near you', icon: Icons.school_rounded),
  _RoleData(label: 'Owner', description: 'Manage your hostel listings and bookings', icon: Icons.business_rounded),
  _RoleData(label: 'Warden', description: 'Monitor residents and daily capacity', icon: Icons.badge_rounded),
  _RoleData(label: 'Parent', description: "Stay updated on your child's accommodation", icon: Icons.people_rounded),
];

class _RoleCard extends StatelessWidget {
  final _RoleData role;
  final bool isSelected;
  final VoidCallback onTap;
  const _RoleCard({required this.role, required this.isSelected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 220),
        curve: Curves.easeOut,
        child: GlassCard(
          borderRadius: 20,
          padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
          tint: isSelected ? AppTheme.tealMint : Colors.white,
          tintOpacity: isSelected ? 0.26 : 0.12,
          border: Border.all(
            color: isSelected
                ? AppTheme.tealMint.withOpacity(0.8)
                : Colors.white.withOpacity(0.2),
            width: isSelected ? 1.6 : 1,
          ),
          child: Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: isSelected
                      ? Colors.white
                      : Colors.white.withOpacity(0.14),
                ),
                child: Icon(
                  role.icon,
                  size: 24,
                  color: isSelected ? AppTheme.teal : Colors.white.withOpacity(0.85),
                ),
              ),
              const SizedBox(width: 14),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      role.label,
                      style: GoogleFonts.dmSans(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      role.description,
                      style: GoogleFonts.dmSans(
                        fontSize: 12,
                        color: Colors.white.withOpacity(0.65),
                      ),
                    ),
                  ],
                ),
              ),
              if (isSelected)
                Container(
                  width: 26,
                  height: 26,
                  decoration: const BoxDecoration(
                    color: AppTheme.tealMint,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(Icons.check_rounded, size: 16, color: Colors.white),
                ),
            ],
          ),
        ),
      ),
    );
  }
}