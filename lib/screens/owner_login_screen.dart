import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';

import 'owner/dashboard/owner_dashboard_screen.dart';
import 'owner_hostel_info_screen.dart';
import 'owner_info_screen.dart';
import 'role_screen.dart';

class OwnerLoginScreen extends StatefulWidget {
  const OwnerLoginScreen({super.key});

  @override
  State<OwnerLoginScreen> createState() => _OwnerLoginScreenState();
}

class _OwnerLoginScreenState extends State<OwnerLoginScreen>
    with SingleTickerProviderStateMixin {
  final _emailCtrl = TextEditingController();
  final _passCtrl = TextEditingController();

  final _emailFocus = FocusNode();
  final _passwordFocus = FocusNode();

  bool _obscure = true;
  bool _loading = false;
  String? _error;

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  @override
  void initState() {
    super.initState();

    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 750),
    );

    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);

    _slide = Tween<double>(
      begin: 28,
      end: 0,
    ).animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));

    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();

    _emailCtrl.dispose();
    _passCtrl.dispose();

    _emailFocus.dispose();
    _passwordFocus.dispose();

    super.dispose();
  }

  // ===========================================================================
  // OWNER LOGIN
  // ===========================================================================

  Future<void> _login() async {
    FocusScope.of(context).unfocus();

    final email = _emailCtrl.text.trim();
    final password = _passCtrl.text;

    if (email.isEmpty || password.isEmpty) {
      setState(() {
        _error = 'Please enter your email and password';
      });
      return;
    }

    final emailPattern = RegExp(
      r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$',
    );

    if (!emailPattern.hasMatch(email)) {
      setState(() {
        _error = 'Please enter a valid email address';
      });
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      // -----------------------------------------------------------------------
      // 1. AUTHENTICATE OWNER
      // -----------------------------------------------------------------------

      final data = await Api().login(email, password);

      final userData = data['user'];

      if (userData is! Map) {
        throw Exception('Invalid user information received from server.');
      }

      final user = Map<String, dynamic>.from(userData);

      final role = user['role']?.toString().trim().toLowerCase();

      // -----------------------------------------------------------------------
      // 2. CHECK ROLE
      // -----------------------------------------------------------------------

      if (role != 'owner' && role != 'admin') {
        await AuthStore.clear();

        if (!mounted) return;

        setState(() {
          _loading = false;
          _error = 'This account is not registered as an owner account.';
        });

        return;
      }

      // -----------------------------------------------------------------------
      // 3. LOAD OWNER HOSTELS
      // -----------------------------------------------------------------------

      final hostelList = await Api().getOwnerHostels();

      if (!mounted) return;

      // -----------------------------------------------------------------------
      // 4. OWNER EXISTS BUT HAS NO HOSTEL
      // -----------------------------------------------------------------------

      if (hostelList.isEmpty) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => OwnerHostelInfoScreen(
              ownerEmail: user['email']?.toString() ?? email,
              ownerName: user['name']?.toString() ?? 'Owner',
            ),
          ),
        );

        return;
      }

      // -----------------------------------------------------------------------
      // 5. OWNER HAS HOSTEL
      // -----------------------------------------------------------------------

      final firstHostel = hostelList.first;

      if (firstHostel is! Map) {
        throw Exception('Invalid hostel information received from server.');
      }

      final hostel = Map<String, dynamic>.from(firstHostel);

      // The owner-hostels endpoint already returns the complete hostel record.
      // Avoid a second detail request here: that endpoint is student-only and
      // previously made the sign-in flow wait for an avoidable failed request.
      final fullHostel = hostel;

      // -----------------------------------------------------------------------
      // 6. OWNER DASHBOARD
      // -----------------------------------------------------------------------

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => OwnerDashboardScreen(
            ownerName: user['name']?.toString() ?? 'Owner',

            ownerEmail: user['email']?.toString() ?? email,

            hostelName: fullHostel['name']?.toString() ?? 'My Hostel',

            hostelType:
                fullHostel['hostel_type']?.toString() ??
                fullHostel['type']?.toString() ??
                'girls',

            city: fullHostel['city']?.toString() ?? '',

            address: fullHostel['address']?.toString() ?? '',
          ),
        ),
      );
    } catch (error) {
      if (!mounted) return;

      await AuthStore.clear();

      setState(() {
        _loading = false;

        _error = error
            .toString()
            .replaceFirst('Exception: ', '')
            .replaceFirst('ClientException: ', '');
      });
    }
  }

  // ===========================================================================
  // REGISTRATION
  // ===========================================================================

  void _openRegistration() {
    FocusScope.of(context).unfocus();

    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const OwnerInfoScreen()),
    );
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    final keyboardHeight = MediaQuery.viewInsetsOf(context).bottom;

    final keyboardOpen = keyboardHeight > 0;

    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset: true,
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.24,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: LayoutBuilder(
              builder: (context, constraints) {
                return SingleChildScrollView(
                  keyboardDismissBehavior:
                      ScrollViewKeyboardDismissBehavior.onDrag,
                  physics: const BouncingScrollPhysics(),
                  padding: EdgeInsets.fromLTRB(
                    18,
                    12,
                    18,
                    keyboardOpen ? keyboardHeight + 16 : 16,
                  ),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(
                      minHeight: keyboardOpen ? 0 : constraints.maxHeight - 28,
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        // =====================================================
                        // BACK BUTTON
                        // =====================================================
                        _buildBackButton(),

                        SizedBox(height: keyboardOpen ? 10 : 17),

                        // =====================================================
                        // WELCOME
                        // =====================================================
                        if (!keyboardOpen) ...[
                          _buildWelcome(),

                          const SizedBox(height: 18),
                        ],

                        // =====================================================
                        // LOGIN CARD
                        // =====================================================
                        AnimatedBuilder(
                          animation: _anim,
                          builder: (context, child) {
                            return Transform.translate(
                              offset: Offset(0, _slide.value),
                              child: Opacity(
                                opacity: _fade.value,
                                child: child,
                              ),
                            );
                          },
                          child: Center(child: _buildLoginCard()),
                        ),

                        if (!keyboardOpen) const SizedBox(height: 16),
                      ],
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

  // ===========================================================================
  // BACK BUTTON
  // ===========================================================================

  void _goBack() {
    FocusScope.of(context).unfocus();
    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => const RoleScreen()),
      (route) => false,
    );
  }

  Widget _buildBackButton() {
    return InkWell(
      onTap: _goBack,
      borderRadius: BorderRadius.circular(50),
      child: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: const Color(0xFF02383B).withOpacity(0.48),
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white.withOpacity(0.25)),
        ),
        child: const Icon(
          Icons.arrow_back_rounded,
          color: Colors.white,
          size: 20,
        ),
      ),
    );
  }

  // ===========================================================================
  // WELCOME
  // ===========================================================================

  Widget _buildWelcome() {
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

          const SizedBox(height: 18),

          Text(
            'Sign in to manage your hostel workspace',
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.74),
              fontSize: 13.5,
              height: 1.4,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // DARK TEAL OWNER LOGIN CARD
  // ===========================================================================

  Widget _buildLoginCard() {
    return Container(
      width: double.infinity,
      constraints: const BoxConstraints(maxWidth: 410),
      padding: const EdgeInsets.fromLTRB(21, 16, 21, 18),
      decoration: BoxDecoration(
        // Dark teal translucent glass
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF06474A).withOpacity(0.84),
            const Color(0xFF02383B).withOpacity(0.80),
            const Color(0xFF01282B).withOpacity(0.84),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(
          color: AppTheme.tealMint.withOpacity(0.25),
          width: 0.9,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.30),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
          BoxShadow(
            color: AppTheme.teal.withOpacity(0.10),
            blurRadius: 20,
            spreadRadius: 1,
          ),
        ],
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ===============================================================
          // HANDLE
          // ===============================================================
          Center(
            child: Container(
              width: 38,
              height: 4,
              margin: const EdgeInsets.only(bottom: 14),
              decoration: BoxDecoration(
                color: AppTheme.tealMint.withOpacity(0.34),
                borderRadius: BorderRadius.circular(3),
              ),
            ),
          ),

          // ===============================================================
          // TITLE
          // ===============================================================
          Row(
            children: [
              Container(
                width: 45,
                height: 45,
                decoration: BoxDecoration(
                  color: const Color(0xFF0C6D70).withOpacity(0.40),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(
                    color: AppTheme.tealMint.withOpacity(0.30),
                  ),
                ),
                child: const Icon(
                  Icons.business_center_outlined,
                  color: AppTheme.tealMint,
                  size: 21,
                ),
              ),

              const SizedBox(width: 12),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Owner Login',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.playfairDisplay(
                        fontSize: 24,
                        height: 1,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),

                    const SizedBox(height: 4),

                    Text(
                      'Access your hostel management workspace',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        fontSize: 10,
                        color: Colors.white.withOpacity(0.58),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 19),

          // ===============================================================
          // EMAIL LABEL
          // ===============================================================
          _label('Email Address'),

          const SizedBox(height: 6),

          // ===============================================================
          // EMAIL FIELD
          // ===============================================================
          _buildField(
            controller: _emailCtrl,
            focusNode: _emailFocus,
            hint: 'Enter your email',
            icon: Icons.mail_outline_rounded,
            keyboardType: TextInputType.emailAddress,
            textInputAction: TextInputAction.next,
            onSubmitted: (_) {
              FocusScope.of(context).requestFocus(_passwordFocus);
            },
          ),

          const SizedBox(height: 13),

          // ===============================================================
          // PASSWORD LABEL
          // ===============================================================
          _label('Password'),

          const SizedBox(height: 6),

          // ===============================================================
          // PASSWORD FIELD
          // ===============================================================
          _buildField(
            controller: _passCtrl,
            focusNode: _passwordFocus,
            hint: 'Enter your password',
            icon: Icons.lock_outline_rounded,
            obscureText: _obscure,
            keyboardType: TextInputType.visiblePassword,
            textInputAction: TextInputAction.done,
            onSubmitted: (_) {
              if (!_loading) {
                _login();
              }
            },
            suffix: IconButton(
              tooltip: _obscure ? 'Show password' : 'Hide password',
              onPressed: () {
                setState(() {
                  _obscure = !_obscure;
                });
              },
              icon: Icon(
                _obscure
                    ? Icons.visibility_outlined
                    : Icons.visibility_off_outlined,
                size: 19,

                // Dark teal icon
                color: const Color(0xFF075D60),
              ),
            ),
          ),

          // ===============================================================
          // ERROR
          // ===============================================================
          if (_error != null) ...[
            const SizedBox(height: 10),
            _buildErrorMessage(),
          ],

          const SizedBox(height: 17),

          // ===============================================================
          // LOGIN BUTTON
          // ===============================================================
          _buildLoginButton(),

          const SizedBox(height: 11),

          // ===============================================================
          // REGISTRATION - kept directly below Sign In so it is always visible
          // ===============================================================
          _buildRegistrationLink(),

          const SizedBox(height: 11),

          // ===============================================================
          // OWNER INFO NOTE
          // ===============================================================
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 9),
            decoration: BoxDecoration(
              color: const Color(0xFF01272A).withOpacity(0.55),
              borderRadius: BorderRadius.circular(13),
              border: Border.all(color: AppTheme.tealMint.withOpacity(0.22)),
            ),
            child: Row(
              children: [
                Container(
                  width: 29,
                  height: 29,
                  decoration: BoxDecoration(
                    color: AppTheme.teal.withOpacity(0.26),
                    borderRadius: BorderRadius.circular(9),
                  ),
                  child: const Icon(
                    Icons.admin_panel_settings_outlined,
                    color: AppTheme.tealMint,
                    size: 16,
                  ),
                ),

                const SizedBox(width: 9),

                Expanded(
                  child: Text(
                    'Secure access to your hostel management account.',
                    style: GoogleFonts.dmSans(
                      color: Colors.white.withOpacity(0.60),
                      fontSize: 9.7,
                      height: 1.35,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // LABEL
  // ===========================================================================

  Widget _label(String text) {
    return Text(
      text,
      style: GoogleFonts.dmSans(
        fontSize: 11.5,
        fontWeight: FontWeight.w700,
        color: Colors.white.withOpacity(0.92),
      ),
    );
  }

  // ===========================================================================
  // WHITE INPUT FIELD
  // ===========================================================================

  Widget _buildField({
    required TextEditingController controller,
    required FocusNode focusNode,
    required String hint,
    required IconData icon,
    bool obscureText = false,
    TextInputType? keyboardType,
    TextInputAction? textInputAction,
    ValueChanged<String>? onSubmitted,
    Widget? suffix,
  }) {
    return AnimatedBuilder(
      animation: focusNode,
      builder: (context, child) {
        final focused = focusNode.hasFocus;

        return AnimatedContainer(
          duration: const Duration(milliseconds: 180),
          height: 52,

          // IMPORTANT:
          // ONE rounded white rectangle only.
          decoration: BoxDecoration(
            color: const Color(0xFFFAFAF8),
            borderRadius: BorderRadius.circular(15),
            border: Border.all(
              color: focused
                  ? const Color(0xFF087A7D)
                  : const Color(0xFFD7D9D5),
              width: focused ? 1.25 : 0.9,
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.08),
                blurRadius: 8,
                offset: const Offset(0, 3),
              ),
            ],
          ),

          child: TextField(
            controller: controller,
            focusNode: focusNode,
            obscureText: obscureText,
            keyboardType: keyboardType,
            textInputAction: textInputAction,
            onSubmitted: onSubmitted,
            autocorrect: false,
            enableSuggestions: !obscureText,

            // Typed input = BLACK
            style: GoogleFonts.dmSans(
              fontSize: 13.5,
              fontWeight: FontWeight.w500,
              color: const Color(0xFF171A1A),
            ),

            cursorColor: const Color(0xFF075D60),

            decoration: InputDecoration(
              hintText: hint,

              // Hint = muted grey
              hintStyle: GoogleFonts.dmSans(
                color: const Color(0xFF8A908D),
                fontSize: 13,
                fontWeight: FontWeight.w400,
              ),

              // Icons remain visible + dark teal
              prefixIcon: Icon(icon, size: 19, color: const Color(0xFF075D60)),

              suffixIcon: suffix,

              // Removes extra internal rectangle.
              filled: false,
              border: InputBorder.none,
              enabledBorder: InputBorder.none,
              focusedBorder: InputBorder.none,
              disabledBorder: InputBorder.none,
              errorBorder: InputBorder.none,
              focusedErrorBorder: InputBorder.none,

              contentPadding: const EdgeInsets.symmetric(
                horizontal: 14,
                vertical: 15,
              ),
            ),
          ),
        );
      },
    );
  }

  // ===========================================================================
  // ERROR MESSAGE
  // ===========================================================================

  Widget _buildErrorMessage() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
      decoration: BoxDecoration(
        color: const Color(0xFF7F2525).withOpacity(0.42),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0xFFFF7777).withOpacity(0.38)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.error_outline_rounded,
            color: Color(0xFFFF8A8A),
            size: 16,
          ),

          const SizedBox(width: 8),

          Expanded(
            child: Text(
              _error!,
              style: GoogleFonts.dmSans(
                color: const Color(0xFFFFB5B5),
                fontSize: 10,
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // LOGIN BUTTON
  // ===========================================================================

  Widget _buildLoginButton() {
    return SizedBox(
      width: double.infinity,
      height: 49,
      child: ElevatedButton(
        onPressed: _loading ? null : _login,
        style: ElevatedButton.styleFrom(
          backgroundColor: AppTheme.teal,
          disabledBackgroundColor: AppTheme.teal.withOpacity(0.50),
          foregroundColor: Colors.white,
          elevation: 0,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
        ),
        child: _loading
            ? const SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  color: Colors.white,
                ),
              )
            : Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(Icons.login_rounded, size: 18),

                  const SizedBox(width: 8),

                  Text(
                    'Sign In',
                    style: GoogleFonts.dmSans(
                      fontSize: 15,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ],
              ),
      ),
    );
  }

  // ===========================================================================
  // REGISTRATION LINK
  // ===========================================================================

  Widget _buildRegistrationLink() {
    return SizedBox(
      width: double.infinity,
      height: 46,
      child: OutlinedButton.icon(
        onPressed: _loading ? null : _openRegistration,
        icon: const Icon(Icons.person_add_alt_1_rounded, size: 17),
        label: Text(
          'New Owner? Register Here',
          style: GoogleFonts.dmSans(
            fontSize: 12.5,
            fontWeight: FontWeight.w800,
          ),
        ),
        style: OutlinedButton.styleFrom(
          foregroundColor: AppTheme.tealMint,
          disabledForegroundColor: AppTheme.tealMint.withOpacity(0.40),
          side: BorderSide(
            color: AppTheme.tealMint.withOpacity(0.42),
            width: 1,
          ),
          backgroundColor: const Color(0xFF01272A).withOpacity(0.34),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
        ),
      ),
    );
  }
}
