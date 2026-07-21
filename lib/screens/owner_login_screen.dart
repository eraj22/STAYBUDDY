import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_info_screen.dart';
import 'owner_dashboard_screen.dart';

class OwnerLoginScreen extends StatefulWidget {
  const OwnerLoginScreen({super.key});

  @override
  State<OwnerLoginScreen> createState() => _OwnerLoginScreenState();
}

class _OwnerLoginScreenState extends State<OwnerLoginScreen>
    with SingleTickerProviderStateMixin {
  final _emailCtrl = TextEditingController();
  final _passCtrl  = TextEditingController();
  bool _obscure    = true;
  bool _loading    = false;
  String? _error;

  late final AnimationController _anim;
  late final Animation<double> _sheetFade;
  late final Animation<double> _sheetSlide;

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(vsync: this, duration: const Duration(milliseconds: 800));
    _sheetFade  = CurvedAnimation(parent: _anim, curve: const Interval(0.2, 1.0, curve: Curves.easeOut));
    _sheetSlide = Tween<double>(begin: 60, end: 0).animate(
      CurvedAnimation(parent: _anim, curve: const Interval(0.2, 1.0, curve: Curves.easeOutCubic)));
    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    _emailCtrl.dispose();
    _passCtrl.dispose();
    super.dispose();
  }

  Future<void> _login() async {
    final email = _emailCtrl.text.trim();
    final pass  = _passCtrl.text.trim();

    if (email.isEmpty || pass.isEmpty) {
      setState(() => _error = 'Please enter email and password');
      return;
    }

    setState(() { _loading = true; _error = null; });

    try {
      final data = await Api().login(email, pass);
      final user = data['user'] as Map<String, dynamic>;

      if (user['role'] != 'owner' && user['role'] != 'admin') {
        await AuthStore.clear();
        setState(() {
          _loading = false;
          _error = 'This account is not an owner account';
        });
        return;
      }

      if (!mounted) return;

      // Go to owner dashboard — fetch hostel info
      final dashboard = await Api().getOwnerDashboard();
      final hostelList = dashboard['hostel_list'] as List? ?? [];

      if (hostelList.isEmpty) {
        // No hostel yet — go to register hostel flow
        Navigator.pushReplacement(context, MaterialPageRoute(
          builder: (_) => const OwnerInfoScreen()));
      } else {
        final h = hostelList[0] as Map<String, dynamic>;
        Navigator.pushReplacement(context, MaterialPageRoute(
          builder: (_) => OwnerDashboardScreen(
            ownerName:  user['name'] ?? 'Owner',
            ownerEmail: user['email'] ?? '',
            hostelName: h['name'] ?? 'My Hostel',
            hostelType: 'girls',
            city:       h['city'] ?? '',
            address:    h['address'] ?? '',
          )));
      }
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = e.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.30,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      GestureDetector(
                        onTap: () => Navigator.pop(context),
                        child: Container(
                          width: 38, height: 38,
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.15),
                            shape: BoxShape.circle,
                            border: Border.all(color: Colors.white.withOpacity(0.3))),
                          child: const Icon(Icons.arrow_back_rounded, color: Colors.white, size: 18)),
                      ),
                      const SizedBox(height: 18),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: Colors.white.withOpacity(0.25))),
                        child: Row(mainAxisSize: MainAxisSize.min, children: [
                          Icon(Icons.verified_rounded, color: AppTheme.tealMint, size: 14),
                          const SizedBox(width: 6),
                          Text('Owner Portal', style: GoogleFonts.dmSans(
                            color: Colors.white.withOpacity(0.9), fontSize: 12, fontWeight: FontWeight.w500)),
                        ]),
                      ),
                      const SizedBox(height: 10),
                      Text('Hello!', style: GoogleFonts.playfairDisplay(
                        fontSize: 38, fontWeight: FontWeight.w700, color: Colors.white)),
                      const SizedBox(height: 4),
                      Text('Welcome back, Owner', style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.6), fontSize: 14)),
                    ],
                  ),
                ),
                const Spacer(),
                AnimatedBuilder(
                  animation: _anim,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, _sheetSlide.value),
                    child: Opacity(opacity: _sheetFade.value, child: child)),
                  child: _buildSheet(),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSheet() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(26, 22, 26, 28),
      decoration: const BoxDecoration(
        color: AppTheme.bg,
        borderRadius: BorderRadius.vertical(top: Radius.circular(32))),
      child: SingleChildScrollView(
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Center(child: Container(width: 40, height: 4, margin: const EdgeInsets.only(bottom: 20),
            decoration: BoxDecoration(color: AppTheme.border, borderRadius: BorderRadius.circular(2)))),

          Row(children: [
            Text('Owner Login', style: GoogleFonts.playfairDisplay(
              fontSize: 24, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
            const SizedBox(width: 10),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(color: AppTheme.teal.withOpacity(0.1), borderRadius: BorderRadius.circular(12)),
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.business_center_rounded, color: AppTheme.teal, size: 12),
                const SizedBox(width: 4),
                Text('Owner', style: GoogleFonts.dmSans(color: AppTheme.teal, fontSize: 11, fontWeight: FontWeight.w600)),
              ]),
            ),
          ]),
          const SizedBox(height: 22),

          _field(ctrl: _emailCtrl, hint: 'Email address', icon: Icons.mail_outline_rounded),
          const SizedBox(height: 12),
          _field(ctrl: _passCtrl, hint: 'Password', icon: Icons.lock_outline_rounded,
            obscure: _obscure,
            suffix: IconButton(
              icon: Icon(_obscure ? Icons.visibility_outlined : Icons.visibility_off_outlined,
                size: 20, color: AppTheme.textMuted),
              onPressed: () => setState(() => _obscure = !_obscure))),

          if (_error != null) ...[
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: Colors.red.shade50,
                borderRadius: BorderRadius.circular(10),
                border: Border.all(color: Colors.red.shade200, width: 0.5)),
              child: Row(children: [
                Icon(Icons.error_outline_rounded, color: Colors.red.shade600, size: 15),
                const SizedBox(width: 8),
                Expanded(child: Text(_error!, style: GoogleFonts.dmSans(
                  fontSize: 12, color: Colors.red.shade700))),
              ]),
            ),
          ],
          const SizedBox(height: 20),

          _primaryBtn('Login', _loading ? null : _login, loading: _loading),
          const SizedBox(height: 20),

          Row(mainAxisAlignment: MainAxisAlignment.center, children: [
            Text("New owner? ", style: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13)),
            GestureDetector(
              onTap: () => Navigator.pushReplacement(context,
                MaterialPageRoute(builder: (_) => const OwnerInfoScreen())),
              child: Text('Register here', style: GoogleFonts.dmSans(
                color: AppTheme.teal, fontSize: 13, fontWeight: FontWeight.w700)),
            ),
          ]),
        ]),
      ),
    );
  }

  Widget _field({required TextEditingController ctrl, required String hint,
      required IconData icon, bool obscure = false, Widget? suffix}) {
    return Container(
      decoration: BoxDecoration(color: AppTheme.card,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border, width: 0.5)),
      child: TextField(
        controller: ctrl, obscureText: obscure,
        style: GoogleFonts.dmSans(fontSize: 14, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 14),
          prefixIcon: Icon(icon, size: 20, color: AppTheme.textMuted),
          suffixIcon: suffix, border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15)),
      ),
    );
  }

  Widget _primaryBtn(String label, VoidCallback? onTap, {bool loading = false}) {
    return SizedBox(
      width: double.infinity, height: 52,
      child: ElevatedButton(
        onPressed: onTap,
        style: ElevatedButton.styleFrom(backgroundColor: AppTheme.teal, foregroundColor: Colors.white,
          elevation: 0, shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
        child: loading
            ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
            : Text(label, style: GoogleFonts.dmSans(fontSize: 16, fontWeight: FontWeight.w600)),
      ),
    );
  }
}
