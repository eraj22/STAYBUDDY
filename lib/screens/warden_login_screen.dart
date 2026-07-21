import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'warden_dashboard_screen.dart';

class WardenLoginScreen extends StatefulWidget {
  const WardenLoginScreen({super.key});

  @override
  State<WardenLoginScreen> createState() => _WardenLoginScreenState();
}

class _WardenLoginScreenState extends State<WardenLoginScreen>
    with SingleTickerProviderStateMixin {
  final _emailCtrl = TextEditingController();
  final _passCtrl  = TextEditingController();
  bool _obscure    = true;
  bool _loading    = false;
  String? _error;

  late final AnimationController _anim;
  late final Animation<double> _fade, _slide;

  @override
  void initState() {
    super.initState();
    _anim  = AnimationController(vsync: this, duration: const Duration(milliseconds: 800));
    _fade  = CurvedAnimation(parent: _anim, curve: Curves.easeOut);
    _slide = Tween<double>(begin: 60, end: 0).animate(
        CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));
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

      if (user['role'] != 'warden' && user['role'] != 'admin') {
        await AuthStore.clear();
        setState(() {
          _loading = false;
          _error = 'This account is not a warden account. Use warden@test.com or fatima@warden.com';
        });
        return;
      }

      if (!mounted) return;

      // Fetch warden dashboard — use defaults if it fails
      String hostelName = 'My Hostel';
      String hostelId   = '1';
      try {
        final dashboard = await Api().getWardenDashboard();
        hostelName = dashboard['hostel_name']?.toString() ?? 'My Hostel';
        hostelId   = dashboard['hostel_id']?.toString() ?? '1';
      } catch (_) {
        // Dashboard fetch failed — proceed with defaults
      }

      if (!mounted) return;
      Navigator.pushReplacement(context, MaterialPageRoute(
        builder: (_) => WardenDashboardScreen(
          wardenName: user['name']?.toString() ?? 'Warden',
          hostelName: hostelName,
          hostelId:   hostelId,
        )));
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
                      const SizedBox(height: 20),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.12),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(color: Colors.white.withOpacity(0.2))),
                        child: Row(mainAxisSize: MainAxisSize.min, children: [
                          const Icon(Icons.badge_rounded, color: Colors.white, size: 14),
                          const SizedBox(width: 6),
                          Text('Warden Portal', style: GoogleFonts.dmSans(
                            color: Colors.white, fontSize: 12, fontWeight: FontWeight.w600)),
                        ]),
                      ),
                      const SizedBox(height: 12),
                      Text('Welcome Back', style: GoogleFonts.playfairDisplay(
                        fontSize: 34, fontWeight: FontWeight.w700, color: Colors.white)),
                      const SizedBox(height: 4),
                      Text('Sign in to manage your hostel', style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.6), fontSize: 14)),
                    ],
                  ),
                ),
                const Spacer(),
                AnimatedBuilder(
                  animation: _anim,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, _slide.value),
                    child: Opacity(opacity: _fade.value, child: child)),
                  child: Container(
                    width: double.infinity,
                    padding: const EdgeInsets.fromLTRB(24, 28, 24, 32),
                    decoration: const BoxDecoration(
                      color: AppTheme.bg,
                      borderRadius: BorderRadius.vertical(top: Radius.circular(32))),
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Center(child: Container(width: 40, height: 4,
                        margin: const EdgeInsets.only(bottom: 24),
                        decoration: BoxDecoration(color: AppTheme.border, borderRadius: BorderRadius.circular(2)))),

                      _label('Email Address'),
                      const SizedBox(height: 8),
                      _field(ctrl: _emailCtrl, hint: 'Enter your email',
                        icon: Icons.email_outlined, type: TextInputType.emailAddress),
                      const SizedBox(height: 16),

                      _label('Password'),
                      const SizedBox(height: 8),
                      _field(
                        ctrl: _passCtrl, hint: 'Enter your password',
                        icon: Icons.lock_outline_rounded, obscure: _obscure,
                        suffix: IconButton(
                          icon: Icon(_obscure ? Icons.visibility_outlined : Icons.visibility_off_outlined,
                            size: 18, color: AppTheme.textMuted),
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
                      const SizedBox(height: 24),

                      SizedBox(
                        width: double.infinity, height: 54,
                        child: ElevatedButton(
                          onPressed: _loading ? null : _login,
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppTheme.teal, foregroundColor: Colors.white,
                            elevation: 0,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                          child: _loading
                              ? const SizedBox(width: 22, height: 22,
                                  child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                              : Text('Sign In', style: GoogleFonts.dmSans(
                                  fontSize: 16, fontWeight: FontWeight.w600)),
                        ),
                      ),
                      const SizedBox(height: 16),
                      Center(child: Text(
                        'Credentials are provided by the hostel owner',
                        style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted))),
                    ]),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _label(String text) => Text(text,
    style: GoogleFonts.dmSans(fontSize: 13, fontWeight: FontWeight.w600, color: AppTheme.textPrimary));

  Widget _field({required TextEditingController ctrl, required String hint,
      required IconData icon, TextInputType? type, bool obscure = false, Widget? suffix}) {
    return Container(
      decoration: BoxDecoration(color: AppTheme.card,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border, width: 0.5)),
      child: TextField(
        controller: ctrl, keyboardType: type, obscureText: obscure,
        style: GoogleFonts.dmSans(fontSize: 14, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13),
          prefixIcon: Icon(icon, size: 18, color: AppTheme.textMuted),
          suffixIcon: suffix, border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16)),
      ),
    );
  }
}