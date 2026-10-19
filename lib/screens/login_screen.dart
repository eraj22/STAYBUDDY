import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../api.dart';
import '../routes.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'discover_hostel_home_page.dart';
import 'student_hostel_access_screen.dart';
import 'my_bookings_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen>
    with SingleTickerProviderStateMixin {
  bool _obscurePassword = true;
  bool _loading = false;
  bool _showResetForm = false;

  final _emailCtrl = TextEditingController();
  final _passwordCtrl = TextEditingController();
  final _resetEmailCtrl = TextEditingController();
  late final AnimationController _controller;
  late final Animation<double> _sheetSlide;
  late final Animation<double> _sheetFade;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );
    _sheetSlide = Tween<double>(begin: 60, end: 0).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.2, 1.0, curve: Curves.easeOutCubic),
      ),
    );
    _sheetFade = CurvedAnimation(
      parent: _controller,
      curve: const Interval(0.2, 1.0, curve: Curves.easeOut),
    );
    _controller.forward();
  }

  @override
  void dispose() {
    _emailCtrl.dispose();
    _passwordCtrl.dispose();
    _resetEmailCtrl.dispose();
    _controller.dispose();
    super.dispose();
  }

  Future<void> _login() async {
    final email = _emailCtrl.text.trim();
    final password = _passwordCtrl.text;

    if (email.isEmpty || password.isEmpty) {
      _showMessage('Please enter email and password.', error: true);
      return;
    }

    setState(() => _loading = true);
    try {
      final data = await Api().login(email, password);

      if (!mounted) return;
      final user = data['user'];
      final role = user is Map ? user['role']?.toString() : null;

      if (role != 'student') {
        await Api().logout();
        throw Exception('Please use a Student account on this login screen.');
      }

      // Owner-created credentials and owner-approved room assignments unlock
      // the internal hostel area. Ordinary student accounts remain in public
      // discovery until a confirmed room assignment exists.
      final access = await Api().getStudentAccess();
      final assignments = access['assignments'];
      final hasInternalAccess = assignments is List && assignments.isNotEmpty;
      final savedBookings = hasInternalAccess ? const <dynamic>[] : await Api().getBookings();
      if (!mounted) return;
      final hasOpenBooking = savedBookings.whereType<Map>().any((booking) {
        final status = booking['status']?.toString().toLowerCase();
        return status == 'pending' || status == 'confirmed';
      });

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => hasInternalAccess
              ? const StudentHostelAccessScreen(openHubImmediately: true)
              : hasOpenBooking
              ? const MyBookingsScreen()
              : const DiscoverHostelHomePage(),
        ),
      );
    } catch (e) {
      if (!mounted) return;
      _showMessage(e.toString().replaceFirst('Exception: ', ''), error: true);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  void _showMessage(String message, {bool error = false}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: error ? Colors.red.shade700 : AppTheme.teal,
      ),
    );
  }

  void _forgotPassword() {
    _resetEmailCtrl.text = _emailCtrl.text.trim();
    setState(() => _showResetForm = true);
  }

  Future<void> _sendResetLink() async {
    final email = _resetEmailCtrl.text.trim();
    if (email.isEmpty) return;
    if (!email.contains('@')) {
      _showMessage('Please enter a valid email address.', error: true);
      return;
    }
    try {
      final result = await Api().requestPasswordReset(email);
      if (!mounted) return;
      setState(() => _showResetForm = false);
      _showMessage(
        result['message']?.toString() ??
            'If an account exists, a reset link has been sent.',
      );
    } catch (error) {
      if (mounted)
        _showMessage(
          error.toString().replaceFirst('Exception: ', ''),
          error: true,
        );
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
          child: Stack(
            children: [
              SafeArea(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // Top greeting
                    Padding(
                      padding: const EdgeInsets.fromLTRB(24, 20, 24, 0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          InkWell(
                            onTap: () {
                              if (Navigator.of(context).canPop()) {
                                Navigator.of(context).pop();
                              } else {
                                Navigator.of(context).pushNamedAndRemoveUntil(
                                  Routes.role,
                                  (route) => false,
                                );
                              }
                            },
                            borderRadius: BorderRadius.circular(50),
                            child: Container(
                              width: 42,
                              height: 42,
                              decoration: BoxDecoration(
                                color: const Color(
                                  0xFF02383B,
                                ).withOpacity(0.48),
                                shape: BoxShape.circle,
                                border: Border.all(
                                  color: Colors.white.withOpacity(0.25),
                                ),
                              ),
                              child: const Icon(
                                Icons.arrow_back_rounded,
                                color: Colors.white,
                              ),
                            ),
                          ),
                          const SizedBox(height: 24),
                          Text(
                            'Welcome\nBack!',
                            style: GoogleFonts.playfairDisplay(
                              fontSize: 40,
                              height: 1.02,
                              fontWeight: FontWeight.w700,
                              color: Colors.white,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Sign in to continue your hostel journey',
                            style: GoogleFonts.dmSans(
                              color: Colors.white.withOpacity(0.6),
                              fontSize: 13,
                            ),
                          ),
                        ],
                      ),
                    ),

                    // Keep the login card directly below the welcome copy instead
                    // of pushing it down to the bottom of the phone frame.
                    const SizedBox(height: 28),

                    // Bottom sheet
                    AnimatedBuilder(
                      animation: _controller,
                      builder: (_, child) => Transform.translate(
                        offset: Offset(0, _sheetSlide.value),
                        child: Opacity(opacity: _sheetFade.value, child: child),
                      ),
                      child: Container(
                        width: double.infinity,
                        padding: const EdgeInsets.fromLTRB(26, 28, 26, 24),
                        decoration: BoxDecoration(
                          gradient: LinearGradient(
                            begin: Alignment.topLeft,
                            end: Alignment.bottomRight,
                            colors: [
                              const Color(0xFF064A4D).withOpacity(0.92),
                              const Color(0xFF02282B).withOpacity(0.94),
                            ],
                          ),
                          borderRadius: BorderRadius.circular(28),
                          border: Border.all(
                            color: AppTheme.tealMint.withOpacity(0.25),
                          ),
                        ),
                        child: SingleChildScrollView(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              // Handle
                              Center(
                                child: Container(
                                  width: 40,
                                  height: 4,
                                  margin: const EdgeInsets.only(bottom: 22),
                                  decoration: BoxDecoration(
                                    color: AppTheme.border,
                                    borderRadius: BorderRadius.circular(2),
                                  ),
                                ),
                              ),

                              Text(
                                'Student Login',
                                style: GoogleFonts.playfairDisplay(
                                  fontSize: 26,
                                  fontWeight: FontWeight.w700,
                                  color: Colors.white,
                                ),
                              ),
                              const SizedBox(height: 22),

                              // Email field
                              _buildField(
                                controller: _emailCtrl,
                                hint: 'Email address',
                                icon: Icons.mail_outline_rounded,
                              ),
                              const SizedBox(height: 12),

                              // Password field
                              _buildField(
                                controller: _passwordCtrl,
                                hint: 'Password',
                                icon: Icons.lock_outline_rounded,
                                obscure: _obscurePassword,
                                suffix: IconButton(
                                  icon: Icon(
                                    _obscurePassword
                                        ? Icons.visibility_outlined
                                        : Icons.visibility_off_outlined,
                                    size: 20,
                                    color: AppTheme.textMuted,
                                  ),
                                  onPressed: () => setState(
                                    () => _obscurePassword = !_obscurePassword,
                                  ),
                                ),
                              ),

                              const SizedBox(height: 10),

                              Align(
                                alignment: Alignment.centerRight,
                                child: TextButton(
                                  onPressed: _forgotPassword,
                                  style: TextButton.styleFrom(
                                    foregroundColor: AppTheme.teal,
                                    padding: EdgeInsets.zero,
                                  ),
                                  child: Text(
                                    'Forgot password?',
                                    style: GoogleFonts.dmSans(
                                      fontSize: 13,
                                      fontWeight: FontWeight.w500,
                                    ),
                                  ),
                                ),
                              ),

                              const SizedBox(height: 8),

                              // Login button
                              SizedBox(
                                width: double.infinity,
                                height: 52,
                                child: ElevatedButton(
                                  onPressed: _loading ? null : _login,
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: AppTheme.teal,
                                    foregroundColor: Colors.white,
                                    elevation: 0,
                                    shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(16),
                                    ),
                                  ),
                                  child: Text(
                                    'Login',
                                    style: GoogleFonts.dmSans(
                                      fontSize: 16,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ),
                              ),

                              const SizedBox(height: 18),

                              // Sign up link
                              Row(
                                mainAxisAlignment: MainAxisAlignment.center,
                                children: [
                                  Text(
                                    "Don't have an account? ",
                                    style: GoogleFonts.dmSans(
                                      color: Colors.white.withOpacity(0.64),
                                      fontSize: 13,
                                    ),
                                  ),
                                  GestureDetector(
                                    onTap: () => Navigator.pushNamed(
                                      context,
                                      Routes.signup,
                                    ),
                                    child: Text(
                                      'Sign up',
                                      style: GoogleFonts.dmSans(
                                        color: AppTheme.tealMint,
                                        fontSize: 13,
                                        fontWeight: FontWeight.w700,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
              if (_showResetForm) _resetPasswordPanel(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _resetPasswordPanel() {
    return Positioned.fill(
      child: ColoredBox(
        color: Colors.black.withOpacity(0.58),
        child: SafeArea(
          child: Center(
            child: SingleChildScrollView(
              padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 24),
              child: Container(
                width: double.infinity,
                padding: const EdgeInsets.all(22),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                    colors: [Color(0xFF07575A), Color(0xFF02383B)],
                  ),
                  borderRadius: BorderRadius.circular(28),
                  border: Border.all(
                    color: AppTheme.tealMint.withOpacity(0.35),
                  ),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Align(
                      alignment: Alignment.centerRight,
                      child: IconButton(
                        onPressed: () => setState(() => _showResetForm = false),
                        icon: const Icon(
                          Icons.close_rounded,
                          color: Colors.white,
                        ),
                      ),
                    ),
                    const Icon(
                      Icons.lock_reset_rounded,
                      color: AppTheme.tealMint,
                      size: 32,
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Reset password',
                      style: GoogleFonts.playfairDisplay(
                        color: Colors.white,
                        fontSize: 28,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      'Enter your student email and we will send a secure reset link.',
                      style: GoogleFonts.dmSans(
                        color: Colors.white70,
                        fontSize: 13,
                      ),
                    ),
                    const SizedBox(height: 22),
                    TextField(
                      controller: _resetEmailCtrl,
                      keyboardType: TextInputType.emailAddress,
                      autofocus: true,
                      style: GoogleFonts.dmSans(color: AppTheme.textPrimary),
                      decoration: InputDecoration(
                        hintText: 'Student email address',
                        prefixIcon: const Icon(
                          Icons.mail_outline_rounded,
                          color: AppTheme.teal,
                        ),
                        filled: true,
                        fillColor: Colors.white,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(16),
                          borderSide: BorderSide.none,
                        ),
                      ),
                    ),
                    const SizedBox(height: 18),
                    SizedBox(
                      width: double.infinity,
                      height: 50,
                      child: ElevatedButton.icon(
                        onPressed: _sendResetLink,
                        icon: const Icon(Icons.send_rounded),
                        label: const Text('Send reset link'),
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppTheme.tealMint,
                          foregroundColor: AppTheme.tealDeep,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(16),
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Center(
                      child: TextButton(
                        onPressed: () => setState(() => _showResetForm = false),
                        child: Text(
                          'Cancel',
                          style: GoogleFonts.dmSans(color: Colors.white70),
                        ),
                      ),
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

  Widget _buildField({
    required TextEditingController controller,
    required String hint,
    required IconData icon,
    bool obscure = false,
    Widget? suffix,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border, width: 0.5),
      ),
      child: TextField(
        controller: controller,
        obscureText: obscure,
        style: GoogleFonts.dmSans(fontSize: 14, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(
            color: AppTheme.textMuted,
            fontSize: 14,
          ),
          prefixIcon: Icon(icon, size: 20, color: AppTheme.textMuted),
          suffixIcon: suffix,
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 16,
            vertical: 15,
          ),
        ),
      ),
    );
  }
}
