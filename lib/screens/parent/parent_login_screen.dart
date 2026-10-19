import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../theme.dart';
import '../../api.dart';
import '../../widgets/responsive_shell.dart';
import '../../widgets/video_background.dart';
import 'parent_dashboard_screen.dart';

class ParentLoginScreen extends StatefulWidget {
  const ParentLoginScreen({super.key});

  @override
  State<ParentLoginScreen> createState() => _ParentLoginScreenState();
}

class _ParentLoginScreenState extends State<ParentLoginScreen>
    with SingleTickerProviderStateMixin {
  final TextEditingController _loginController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;
  late Animation<Offset> _slideAnimation;

  bool _obscurePassword = true;
  bool _loading = false;

  static const Color _background = Color(0xFF063E40);
  static const Color _darkTeal = Color(0xFF074F51);
  static const Color _teal = Color(0xFF0A6B6E);
  static const Color _mint = Color(0xFF5DCAA5);

  @override
  void initState() {
    super.initState();

    _animationController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );

    _fadeAnimation = CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeOut,
    );

    _slideAnimation =
        Tween<Offset>(begin: const Offset(0, 0.08), end: Offset.zero).animate(
          CurvedAnimation(
            parent: _animationController,
            curve: Curves.easeOutCubic,
          ),
        );

    _animationController.forward();
  }

  @override
  void dispose() {
    _loginController.dispose();
    _passwordController.dispose();
    _animationController.dispose();
    super.dispose();
  }

  Future<void> _login() async {
    if (_loginController.text.trim().isEmpty ||
        _passwordController.text.trim().isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Please enter your login ID and password.'),
        ),
      );

      return;
    }

    setState(() {
      _loading = true;
    });

    try {
      final result = await Api().login(
        _loginController.text.trim(),
        _passwordController.text,
      );
      final user = Map<String, dynamic>.from(result['user'] as Map);
      if (user['role']?.toString().toLowerCase() != 'parent') {
        await AuthStore.clear();
        throw Exception('This account is not a parent account.');
      }
      if (!mounted) return;
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (_) => const ParentDashboardScreen()),
      );
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(error.toString().replaceFirst('Exception: ', '')),
        ),
      );
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset: true,
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: .24,
          child: SafeArea(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final keyboardOpen =
                    MediaQuery.viewInsetsOf(context).bottom > 0;
                return SingleChildScrollView(
                  physics: const BouncingScrollPhysics(),
                  keyboardDismissBehavior:
                      ScrollViewKeyboardDismissBehavior.onDrag,
                  padding: const EdgeInsets.fromLTRB(18, 12, 18, 16),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(
                      minHeight: constraints.maxHeight - 28,
                    ),
                    child: FadeTransition(
                      opacity: _fadeAnimation,
                      child: SlideTransition(
                        position: _slideAnimation,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            _buildBackButton(),
                            SizedBox(height: keyboardOpen ? 8 : 16),
                            if (!keyboardOpen) _buildWardenStyleWelcome(),
                            SizedBox(height: keyboardOpen ? 8 : 18),
                            Center(child: _buildLoginCard()),
                          ],
                        ),
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildBackButton() {
    return InkWell(
      onTap: () => Navigator.maybePop(context),
      borderRadius: BorderRadius.circular(50),
      child: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: const Color(0xFF02383B).withOpacity(.48),
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white.withOpacity(.25)),
        ),
        child: const Icon(
          Icons.arrow_back_rounded,
          color: Colors.white,
          size: 20,
        ),
      ),
    );
  }

  Widget _buildWardenStyleWelcome() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 3),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Welcome\nBack!',
            style: GoogleFonts.playfairDisplay(
              fontSize: 40,
              height: 1.02,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 20),
          Text(
            'Sign in to stay connected with your child',
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(.74),
              fontSize: 13.5,
              height: 1.4,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLoginContent() {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        _buildBranding(),

        const SizedBox(height: 30),

        _buildLoginCard(),

        const SizedBox(height: 18),

        Text(
          'Secure access for registered StayBuddy parents',
          textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(color: Colors.white38, fontSize: 11),
        ),
      ],
    );
  }

  Widget _buildBranding() {
    return Column(
      children: [
        Container(
          width: 78,
          height: 78,
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            color: _teal.withOpacity(.12),
            border: Border.all(color: _mint.withOpacity(.5), width: 1.5),
            boxShadow: [
              BoxShadow(
                color: _teal.withOpacity(.18),
                blurRadius: 30,
                spreadRadius: 5,
              ),
            ],
          ),
          child: const Icon(
            Icons.family_restroom_rounded,
            color: _mint,
            size: 38,
          ),
        ),

        const SizedBox(height: 18),

        Text(
          'StayBuddy',
          style: GoogleFonts.playfairDisplay(
            color: Colors.white,
            fontSize: 34,
            fontWeight: FontWeight.w700,
          ),
        ),

        const SizedBox(height: 5),

        Text(
          'Parent Portal',
          style: GoogleFonts.dmSans(
            color: _mint,
            fontSize: 14,
            fontWeight: FontWeight.w600,
            letterSpacing: 1,
          ),
        ),
      ],
    );
  }

  Widget _buildLoginCard() {
    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(maxWidth: 410),
      padding: const EdgeInsets.fromLTRB(21, 16, 21, 18),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF064A4D).withOpacity(.88),
            const Color(0xFF033A3D).withOpacity(.86),
            const Color(0xFF02282B).withOpacity(.90),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(
          color: AppTheme.tealMint.withOpacity(.25),
          width: .9,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(.30),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 45,
                height: 45,
                decoration: BoxDecoration(
                  color: const Color(0xFF0C6D70).withOpacity(.38),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: _mint.withOpacity(.30)),
                ),
                child: const Icon(
                  Icons.family_restroom_outlined,
                  color: _mint,
                  size: 21,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Parent Login',
                      style: GoogleFonts.playfairDisplay(
                        fontSize: 24,
                        height: 1,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Access your child care workspace',
                      style: GoogleFonts.dmSans(
                        fontSize: 10,
                        color: Colors.white.withOpacity(.58),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 18),

          _buildLabel('Login ID / Email'),

          const SizedBox(height: 8),

          _buildTextField(
            controller: _loginController,
            hint: 'Enter your parent login ID',
            icon: Icons.person_outline_rounded,
          ),

          const SizedBox(height: 18),

          _buildLabel('Password'),

          const SizedBox(height: 8),

          _buildTextField(
            controller: _passwordController,
            hint: 'Enter your password',
            icon: Icons.lock_outline_rounded,
            obscure: _obscurePassword,
            suffix: IconButton(
              onPressed: () {
                setState(() {
                  _obscurePassword = !_obscurePassword;
                });
              },
              icon: Icon(
                _obscurePassword
                    ? Icons.visibility_off_outlined
                    : Icons.visibility_outlined,
                color: const Color(0xFF087D80),
                size: 20,
              ),
            ),
          ),

          const SizedBox(height: 20),

          SizedBox(
            width: double.infinity,
            height: 54,
            child: ElevatedButton(
              onPressed: _loading ? null : _login,
              style: ElevatedButton.styleFrom(
                backgroundColor: _teal,
                disabledBackgroundColor: _teal.withOpacity(.5),
                foregroundColor: Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(16),
                ),
              ),
              child: _loading
                  ? const SizedBox(
                      width: 22,
                      height: 22,
                      child: CircularProgressIndicator(
                        strokeWidth: 2.5,
                        color: Colors.white,
                      ),
                    )
                  : Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Text(
                          'Sign In',
                          style: GoogleFonts.dmSans(
                            fontSize: 14,
                            fontWeight: FontWeight.w700,
                          ),
                        ),
                        const SizedBox(width: 8),
                        const Icon(Icons.arrow_forward_rounded, size: 19),
                      ],
                    ),
            ),
          ),

          const SizedBox(height: 22),

          _buildCredentialInformation(),
        ],
      ),
    );
  }

  Widget _buildLabel(String text) {
    return Text(
      text,
      style: GoogleFonts.dmSans(
        color: Colors.white70,
        fontSize: 12,
        fontWeight: FontWeight.w600,
      ),
    );
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String hint,
    required IconData icon,
    bool obscure = false,
    Widget? suffix,
  }) {
    return TextField(
      controller: controller,
      obscureText: obscure,
      style: GoogleFonts.dmSans(color: const Color(0xFF161616), fontSize: 13),
      cursorColor: const Color(0xFF087D80),
      decoration: InputDecoration(
        hintText: hint,
        hintStyle: GoogleFonts.dmSans(
          color: const Color(0xFF979995),
          fontSize: 12,
        ),
        prefixIcon: Icon(icon, color: const Color(0xFF087D80), size: 20),
        suffixIcon: suffix,
        filled: true,
        fillColor: const Color(0xFFF8F8F6),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 16,
          vertical: 17,
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: BorderSide(color: const Color(0xFFD7D8D3)),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: _teal, width: 1.4),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(16),
          borderSide: const BorderSide(color: Colors.redAccent),
        ),
      ),
    );
  }

  Widget _buildCredentialInformation() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _mint.withOpacity(.055),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _mint.withOpacity(.14)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: _teal.withOpacity(.12),
              borderRadius: BorderRadius.circular(11),
            ),
            child: const Icon(
              Icons.admin_panel_settings_outlined,
              color: _mint,
              size: 20,
            ),
          ),

          const SizedBox(width: 11),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Owner-generated credentials',
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                  ),
                ),

                const SizedBox(height: 4),

                Text(
                  'Your hostel owner provides your login ID and password. '
                  'Contact hostel administration if you have not received them.',
                  style: GoogleFonts.dmSans(
                    color: Colors.white54,
                    fontSize: 10.5,
                    height: 1.45,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
