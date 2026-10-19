import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'role_screen.dart';
import 'warden_dashboard_screen.dart';

class WardenLoginScreen extends StatefulWidget {
  const WardenLoginScreen({super.key});

  @override
  State<WardenLoginScreen> createState() =>
      _WardenLoginScreenState();
}

class _WardenLoginScreenState
    extends State<WardenLoginScreen>
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
      duration: const Duration(
        milliseconds: 750,
      ),
    );

    _fade = CurvedAnimation(
      parent: _anim,
      curve: Curves.easeOut,
    );

    _slide = Tween<double>(
      begin: 28,
      end: 0,
    ).animate(
      CurvedAnimation(
        parent: _anim,
        curve: Curves.easeOutCubic,
      ),
    );

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
  // LOGIN
  // ===========================================================================

  Future<void> _login() async {
    FocusScope.of(context).unfocus();

    final email = _emailCtrl.text.trim();
    final pass = _passCtrl.text.trim();

    if (email.isEmpty || pass.isEmpty) {
      setState(() {
        _error =
            'Please enter your email and password';
      });
      return;
    }

    final emailPattern = RegExp(
      r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$',
    );

    if (!emailPattern.hasMatch(email)) {
      setState(() {
        _error =
            'Please enter a valid email address';
      });
      return;
    }

    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      final data = await Api().login(
        email,
        pass,
      );

      final rawUser = data['user'];

      if (rawUser is! Map) {
        throw Exception(
          'Invalid user information received',
        );
      }

      final user =
          Map<String, dynamic>.from(
        rawUser,
      );

      final role = user['role']
          ?.toString()
          .toLowerCase();

      if (role != 'warden' &&
          role != 'admin') {
        await AuthStore.clear();

        if (!mounted) return;

        setState(() {
          _loading = false;

          _error =
              'This account is not registered as a warden account';
        });

        return;
      }

      String hostelName = 'My Hostel';
      String hostelId = '1';

      try {
        final dashboard =
            await Api().getWardenDashboard();

        final fetchedHostelName =
            dashboard['hostel_name']
                ?.toString()
                .trim();

        final fetchedHostelId =
            dashboard['hostel_id']
                ?.toString()
                .trim();

        if (fetchedHostelName != null &&
            fetchedHostelName.isNotEmpty) {
          hostelName =
              fetchedHostelName;
        }

        if (fetchedHostelId != null &&
            fetchedHostelId.isNotEmpty) {
          hostelId =
              fetchedHostelId;
        }
      } catch (e) {
        debugPrint(
          'Warden dashboard fetch failed: $e',
        );
      }

      if (!mounted) return;

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) =>
              WardenDashboardScreen(
            wardenName:
                user['name']?.toString() ??
                    'Warden',
            hostelName: hostelName,
            hostelId: hostelId,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _loading = false;

        _error = e
            .toString()
            .replaceFirst(
              'Exception: ',
              '',
            )
            .replaceFirst(
              'ClientException: ',
              '',
            );
      });
    }
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset: true,
        backgroundColor:
            Colors.transparent,
        body: VideoBackground(
          assetPath:
              'assets/images/background.mp4',
          overlayOpacity: 0.24,
          cardMargin:
              const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: LayoutBuilder(
              builder: (
                context,
                constraints,
              ) {
                final keyboardOpen =
                    MediaQuery.viewInsetsOf(
                          context,
                        ).bottom >
                        0;

                return Padding(
                  padding:
                      const EdgeInsets.fromLTRB(
                    18,
                    12,
                    18,
                    16,
                  ),
                  child: Column(
                    crossAxisAlignment:
                        CrossAxisAlignment.start,
                    children: [
                      // =======================================================
                      // BACK BUTTON
                      // =======================================================

                      _buildBackButton(),

                      SizedBox(
                        height:
                            keyboardOpen
                                ? 8
                                : 16,
                      ),

                      // =======================================================
                      // WELCOME
                      // =======================================================

                      if (!keyboardOpen)
                        _buildWelcome(),

                      SizedBox(
                        height:
                            keyboardOpen
                                ? 8
                                : 14,
                      ),

                      // =======================================================
                      // LOGIN CARD
                      // =======================================================

                      Expanded(
                        child: Align(
                          alignment:
                              Alignment.topCenter,
                          child:
                              AnimatedBuilder(
                            animation: _anim,
                            builder: (
                              context,
                              child,
                            ) {
                              return Transform
                                  .translate(
                                offset: Offset(
                                  0,
                                  _slide.value,
                                ),
                                child: Opacity(
                                  opacity:
                                      _fade.value,
                                  child: child,
                                ),
                              );
                            },
                            child: Padding(
                              padding:
                                  const EdgeInsets
                                      .only(
                                top: 4,
                              ),
                              child:
                                  _buildLoginCard(),
                            ),
                          ),
                        ),
                      ),
                    ],
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
    if (Navigator.of(context).canPop()) {
      Navigator.of(context).pop();
      return;
    }

    Navigator.of(context).pushAndRemoveUntil(
      MaterialPageRoute(builder: (_) => const RoleScreen()),
      (route) => false,
    );
  }

  Widget _buildBackButton() {
    return InkWell(
      onTap: _goBack,
      borderRadius:
          BorderRadius.circular(50),
      child: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: const Color(
            0xFF02383B,
          ).withOpacity(
            0.48,
          ),
          shape: BoxShape.circle,
          border: Border.all(
            color:
                Colors.white.withOpacity(
              0.25,
            ),
          ),
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
      padding:
          const EdgeInsets.symmetric(
        horizontal: 3,
      ),
      child: Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          Text(
            'Welcome\nBack!',
            style:
                GoogleFonts.playfairDisplay(
              fontSize: 40,
              height: 1.02,
              fontWeight:
                  FontWeight.w700,
              color: Colors.white,
            ),
          ),

          // One-line gap
          const SizedBox(
            height: 20,
          ),

          Text(
            'Sign in to manage your assigned hostel',
            style:
                GoogleFonts.dmSans(
              color:
                  Colors.white.withOpacity(
                0.74,
              ),
              fontSize: 13.5,
              height: 1.4,
              fontWeight:
                  FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // DARK TEAL LOGIN CARD
  // ===========================================================================

  Widget _buildLoginCard() {
    return Container(
      width: double.infinity,

      constraints:
          const BoxConstraints(
        maxWidth: 410,
      ),

      padding:
          const EdgeInsets.fromLTRB(
        21,
        16,
        21,
        18,
      ),

      decoration: BoxDecoration(
        // Dark teal transparent glass
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(
              0xFF064A4D,
            ).withOpacity(
              0.88,
            ),
            const Color(
              0xFF033A3D,
            ).withOpacity(
              0.86,
            ),
            const Color(
              0xFF02282B,
            ).withOpacity(
              0.90,
            ),
          ],
        ),

        borderRadius:
            BorderRadius.circular(28),

        border: Border.all(
          color: AppTheme.tealMint
              .withOpacity(
            0.25,
          ),
          width: 0.9,
        ),

        boxShadow: [
          BoxShadow(
            color:
                Colors.black.withOpacity(
              0.30,
            ),
            blurRadius: 24,
            offset:
                const Offset(
              0,
              10,
            ),
          ),
          BoxShadow(
            color:
                AppTheme.teal.withOpacity(
              0.10,
            ),
            blurRadius: 20,
            spreadRadius: 1,
          ),
        ],
      ),

      child: Column(
        mainAxisSize:
            MainAxisSize.min,
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          // ===============================================================
          // HANDLE
          // ===============================================================

          Center(
            child: Container(
              width: 38,
              height: 4,
              margin:
                  const EdgeInsets.only(
                bottom: 14,
              ),
              decoration:
                  BoxDecoration(
                color: AppTheme.tealMint
                    .withOpacity(
                  0.32,
                ),
                borderRadius:
                    BorderRadius.circular(
                  3,
                ),
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
                decoration:
                    BoxDecoration(
                  color: const Color(
                    0xFF0C6D70,
                  ).withOpacity(
                    0.38,
                  ),
                  borderRadius:
                      BorderRadius.circular(
                    14,
                  ),
                  border: Border.all(
                    color: AppTheme
                        .tealMint
                        .withOpacity(
                      0.30,
                    ),
                  ),
                ),
                child: const Icon(
                  Icons.badge_outlined,
                  color:
                      AppTheme.tealMint,
                  size: 21,
                ),
              ),

              const SizedBox(
                width: 12,
              ),

              Expanded(
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Warden Login',
                      style: GoogleFonts
                          .playfairDisplay(
                        fontSize: 24,
                        height: 1,
                        fontWeight:
                            FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),

                    const SizedBox(
                      height: 4,
                    ),

                    Text(
                      'Access your hostel workspace',
                      style:
                          GoogleFonts.dmSans(
                        fontSize: 10,
                        color: Colors.white
                            .withOpacity(
                          0.58,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(
            height: 18,
          ),

          // ===============================================================
          // EMAIL
          // ===============================================================

          _label(
            'Email Address',
          ),

          const SizedBox(
            height: 6,
          ),

          _field(
            ctrl: _emailCtrl,
            focusNode:
                _emailFocus,
            hint:
                'Enter your email',
            icon:
                Icons.mail_outline_rounded,
            type:
                TextInputType.emailAddress,
            action:
                TextInputAction.next,
            onSubmitted: (_) {
              FocusScope.of(
                context,
              ).requestFocus(
                _passwordFocus,
              );
            },
          ),

          const SizedBox(
            height: 12,
          ),

          // ===============================================================
          // PASSWORD
          // ===============================================================

          _label(
            'Password',
          ),

          const SizedBox(
            height: 6,
          ),

          _field(
            ctrl: _passCtrl,
            focusNode:
                _passwordFocus,
            hint:
                'Enter your password',
            icon:
                Icons.lock_outline_rounded,
            obscure:
                _obscure,
            action:
                TextInputAction.done,
            onSubmitted: (_) {
              if (!_loading) {
                _login();
              }
            },

            // Eye icon dark teal
            suffix: IconButton(
              tooltip:
                  _obscure
                      ? 'Show password'
                      : 'Hide password',
              icon: Icon(
                _obscure
                    ? Icons
                        .visibility_outlined
                    : Icons
                        .visibility_off_outlined,
                size: 19,
                color: const Color(
                  0xFF00666A,
                ),
              ),
              onPressed: () {
                setState(() {
                  _obscure =
                      !_obscure;
                });
              },
            ),
          ),

          // ===============================================================
          // ERROR
          // ===============================================================

          if (_error != null) ...[
            const SizedBox(
              height: 10,
            ),
            _buildError(),
          ],

          const SizedBox(
            height: 17,
          ),

          // ===============================================================
          // SIGN IN BUTTON
          // ===============================================================

          SizedBox(
            width: double.infinity,
            height: 49,
            child: ElevatedButton(
              onPressed:
                  _loading
                      ? null
                      : _login,
              style:
                  ElevatedButton.styleFrom(
                backgroundColor:
                    AppTheme.teal,
                disabledBackgroundColor:
                    AppTheme.teal
                        .withOpacity(
                  0.50,
                ),
                foregroundColor:
                    Colors.white,
                elevation: 0,
                shape:
                    RoundedRectangleBorder(
                  borderRadius:
                      BorderRadius.circular(
                    15,
                  ),
                ),
              ),
              child:
                  _loading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child:
                              CircularProgressIndicator(
                            strokeWidth: 2,
                            color:
                                Colors.white,
                          ),
                        )
                      : Row(
                          mainAxisAlignment:
                              MainAxisAlignment
                                  .center,
                          children: [
                            const Icon(
                              Icons
                                  .login_rounded,
                              size: 18,
                            ),

                            const SizedBox(
                              width: 8,
                            ),

                            Text(
                              'Sign In',
                              style:
                                  GoogleFonts.dmSans(
                                fontSize: 15,
                                fontWeight:
                                    FontWeight
                                        .w800,
                              ),
                            ),
                          ],
                        ),
            ),
          ),

          const SizedBox(
            height: 13,
          ),

          // ===============================================================
          // LOGIN NOTE
          // ===============================================================

          Container(
            width: double.infinity,
            padding:
                const EdgeInsets.symmetric(
              horizontal: 11,
              vertical: 9,
            ),
            decoration:
                BoxDecoration(
              color: const Color(
                0xFF01272A,
              ).withOpacity(
                0.55,
              ),
              borderRadius:
                  BorderRadius.circular(
                13,
              ),
              border: Border.all(
                color: AppTheme
                    .tealMint
                    .withOpacity(
                  0.22,
                ),
              ),
            ),
            child: Row(
              children: [
                Container(
                  width: 29,
                  height: 29,
                  decoration:
                      BoxDecoration(
                    color: AppTheme.teal
                        .withOpacity(
                      0.26,
                    ),
                    borderRadius:
                        BorderRadius.circular(
                      9,
                    ),
                  ),
                  child: const Icon(
                    Icons
                        .verified_user_outlined,
                    color:
                        AppTheme.tealMint,
                    size: 16,
                  ),
                ),

                const SizedBox(
                  width: 9,
                ),

                Expanded(
                  child: Text(
                    'Login credentials are provided by your hostel owner.',
                    style:
                        GoogleFonts.dmSans(
                      color: Colors.white
                          .withOpacity(
                        0.60,
                      ),
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

  Widget _label(
    String text,
  ) {
    return Text(
      text,
      style:
          GoogleFonts.dmSans(
        fontSize: 11.5,
        fontWeight:
            FontWeight.w700,
        color:
            Colors.white.withOpacity(
          0.92,
        ),
      ),
    );
  }

  // ===========================================================================
  // WHITE INPUT FIELD
  // ===========================================================================

 Widget _field({
  required TextEditingController ctrl,
  required FocusNode focusNode,
  required String hint,
  required IconData icon,
  TextInputType? type,
  TextInputAction? action,
  bool obscure = false,
  Widget? suffix,
  ValueChanged<String>? onSubmitted,
}) {
  return AnimatedBuilder(
    animation: focusNode,
    builder: (context, child) {
      final focused = focusNode.hasFocus;

      return AnimatedContainer(
        duration: const Duration(
          milliseconds: 180,
        ),
        height: 56,

        decoration: BoxDecoration(
          // CLEAN WHITE ROUNDED FIELD
          color: const Color(0xFFF8F8F6),

          borderRadius: BorderRadius.circular(
            17,
          ),

          border: Border.all(
            color: focused
                ? const Color(0xFF087D80)
                : const Color(0xFFD7D8D3),
            width: focused ? 1.5 : 1.0,
          ),

          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(
                0.08,
              ),
              blurRadius: 8,
              offset: const Offset(
                0,
                3,
              ),
            ),
          ],
        ),

        child: TextField(
          controller: ctrl,
          focusNode: focusNode,
          keyboardType: type,
          textInputAction: action,
          obscureText: obscure,
          onSubmitted: onSubmitted,
          autocorrect: false,
          enableSuggestions: !obscure,

          // ============================================================
          // TYPED TEXT = BLACK
          // ============================================================

          style: GoogleFonts.dmSans(
            fontSize: 13.5,
            fontWeight: FontWeight.w500,
            color: const Color(
              0xFF161616,
            ),
          ),

          cursorColor: const Color(
            0xFF087D80,
          ),

          decoration: InputDecoration(
            hintText: hint,

            // ==========================================================
            // HINT TEXT = GREY
            // ==========================================================

            hintStyle: GoogleFonts.dmSans(
              color: const Color(
                0xFF979995,
              ),
              fontSize: 13,
              fontWeight: FontWeight.w400,
            ),

            // ==========================================================
            // LEFT ICON = DARK TEAL
            // ==========================================================

            prefixIcon: Icon(
              icon,
              size: 20,
              color: const Color(
                0xFF087D80,
              ),
            ),

            // Password eye icon
            suffixIcon: suffix,

            filled: false,

            border: InputBorder.none,
            enabledBorder: InputBorder.none,
            focusedBorder: InputBorder.none,
            disabledBorder: InputBorder.none,
            errorBorder: InputBorder.none,

            contentPadding:
                const EdgeInsets.symmetric(
              horizontal: 14,
              vertical: 17,
            ),
          ),
        ),
      );
    },
  );
}
  // ===========================================================================
  // ERROR
  // ===========================================================================

  Widget _buildError() {
    return Container(
      width:
          double.infinity,

      padding:
          const EdgeInsets.symmetric(
        horizontal: 12,
        vertical: 9,
      ),

      decoration:
          BoxDecoration(
        color: const Color(
          0xFF7F2525,
        ).withOpacity(
          0.42,
        ),

        borderRadius:
            BorderRadius.circular(
          12,
        ),

        border: Border.all(
          color: const Color(
            0xFFFF7777,
          ).withOpacity(
            0.38,
          ),
        ),
      ),

      child: Row(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons
                .error_outline_rounded,
            color: Color(
              0xFFFF8A8A,
            ),
            size: 16,
          ),

          const SizedBox(
            width: 8,
          ),

          Expanded(
            child: Text(
              _error!,
              style:
                  GoogleFonts.dmSans(
                color:
                    const Color(
                  0xFFFFB5B5,
                ),
                fontSize: 10,
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
