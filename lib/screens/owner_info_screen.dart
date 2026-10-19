import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_otp_screen.dart';

class OwnerInfoScreen extends StatefulWidget {
  const OwnerInfoScreen({super.key});

  @override
  State<OwnerInfoScreen> createState() =>
      _OwnerInfoScreenState();
}

class _OwnerInfoScreenState extends State<OwnerInfoScreen>
    with SingleTickerProviderStateMixin {
  final _nameCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _cnicCtrl = TextEditingController();
  final _cityCtrl = TextEditingController();
  final _phoneCtrl = TextEditingController();
  final _passwordCtrl = TextEditingController();
  final _confirmPasswordCtrl = TextEditingController();

  final _nameFocus = FocusNode();
  final _emailFocus = FocusNode();
  final _cnicFocus = FocusNode();
  final _cityFocus = FocusNode();
  final _phoneFocus = FocusNode();
  final _passwordFocus = FocusNode();
  final _confirmPasswordFocus = FocusNode();

  bool _obscurePassword = true;
  bool _obscureConfirmPassword = true;

  String? _error;

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  @override
  void initState() {
    super.initState();

    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 800),
    );

    _fade = CurvedAnimation(
      parent: _anim,
      curve: Curves.easeOut,
    );

    _slide = Tween<double>(
      begin: 34,
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

    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _cnicCtrl.dispose();
    _cityCtrl.dispose();
    _phoneCtrl.dispose();
    _passwordCtrl.dispose();
    _confirmPasswordCtrl.dispose();

    _nameFocus.dispose();
    _emailFocus.dispose();
    _cnicFocus.dispose();
    _cityFocus.dispose();
    _phoneFocus.dispose();
    _passwordFocus.dispose();
    _confirmPasswordFocus.dispose();

    super.dispose();
  }

  // ===========================================================================
  // VALIDATION + CONTINUE
  // ===========================================================================

  Future<void> _sendOtp() async {
    FocusScope.of(context).unfocus();

    final name = _nameCtrl.text.trim();
    final email = _emailCtrl.text.trim().toLowerCase();
    final cnic = _cnicCtrl.text.trim();
    final city = _cityCtrl.text.trim();
    final phone = _phoneCtrl.text.trim();
    final password = _passwordCtrl.text;
    final confirmPassword = _confirmPasswordCtrl.text;

    if ([
      name,
      email,
      cnic,
      city,
      phone,
      password,
      confirmPassword,
    ].any((value) => value.isEmpty)) {
      setState(() {
        _error = 'Please fill all fields';
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

    if (cnic.length != 13) {
      setState(() {
        _error = 'CNIC must contain exactly 13 digits';
      });
      return;
    }

    if (phone.length != 11 || !phone.startsWith('03')) {
      setState(() {
        _error = 'Phone number must be 11 digits and start with 03';
      });
      return;
    }

    if (password.length < 8) {
      setState(() {
        _error = 'Password must be at least 8 characters';
      });
      return;
    }

    if (!RegExp(r'[A-Z]').hasMatch(password)) {
      setState(() {
        _error = 'Password must contain at least one uppercase letter';
      });
      return;
    }

    if (!RegExp(r'[a-z]').hasMatch(password)) {
      setState(() {
        _error = 'Password must contain at least one lowercase letter';
      });
      return;
    }

    if (!RegExp(r'[0-9]').hasMatch(password)) {
      setState(() {
        _error = 'Password must contain at least one number';
      });
      return;
    }

    if (password != confirmPassword) {
      setState(() {
        _error = 'Passwords do not match';
      });
      return;
    }

    setState(() {
      _error = null;
    });

    try {
      await Api().sendOwnerOtp(
        name: name,
        email: email,
        password: password,
        cnic: cnic,
        city: city,
        phone: phone,
      );

      if (!mounted) return;

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => OwnerOtpScreen(
            email: email,
            phone: phone,
            name: name,
            cnic: cnic,
            city: city,
            password: password,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;

      final message = e
          .toString()
          .replaceFirst('Exception: ', '')
          .trim();

      setState(() {
        _error = message.isEmpty
            ? 'Unable to send OTP. Please try again.'
            : message;
      });
    }
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    final keyboardHeight =
        MediaQuery.viewInsetsOf(context).bottom;

    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset: true,
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath:
              'assets/images/background.mp4',
          overlayOpacity: 0.25,
          cardMargin:
              const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: LayoutBuilder(
              builder: (
                context,
                constraints,
              ) {
                return SingleChildScrollView(
                  physics:
                      const BouncingScrollPhysics(),
                  keyboardDismissBehavior:
                      ScrollViewKeyboardDismissBehavior
                          .onDrag,
                  padding: EdgeInsets.fromLTRB(
                    18,
                    12,
                    18,
                    24 + keyboardHeight,
                  ),
                  child: ConstrainedBox(
                    constraints: BoxConstraints(
                      minHeight:
                          constraints.maxHeight - 36,
                    ),
                    child: Column(
                      crossAxisAlignment:
                          CrossAxisAlignment.start,
                      children: [
                        _buildHeader(),

                        const SizedBox(height: 18),

                        AnimatedBuilder(
                          animation: _anim,
                          builder: (
                            context,
                            child,
                          ) {
                            return Transform.translate(
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
                          child:
                              _buildProfileCard(),
                        ),

                        const SizedBox(height: 24),
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
  // HEADER
  // ===========================================================================

  Widget _buildHeader() {
    return Padding(
      padding:
          const EdgeInsets.symmetric(
        horizontal: 3,
      ),
      child: Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          InkWell(
            onTap: () =>
                Navigator.maybePop(context),
            borderRadius:
                BorderRadius.circular(50),
            child: Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                color:
                    const Color(0xFF02383B)
                        .withOpacity(0.48),
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
          ),

          const SizedBox(height: 16),

          const _StepIndicator(
            step: 1,
            total: 4,
          ),

          const SizedBox(height: 18),

          Text(
            'Create Your\nProfile',
            style:
                GoogleFonts.playfairDisplay(
              fontSize: 37,
              height: 1.05,
              fontWeight:
                  FontWeight.w700,
              color: Colors.white,
            ),
          ),

          const SizedBox(height: 12),

          Text(
            'Tell us a little about yourself to set up your owner account.',
            style:
                GoogleFonts.dmSans(
              color:
                  Colors.white.withOpacity(
                0.72,
              ),
              fontSize: 12.5,
              height: 1.45,
              fontWeight:
                  FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // MAIN GLASS CARD
  // ===========================================================================

  Widget _buildProfileCard() {
    return Container(
      width: double.infinity,
      constraints:
          const BoxConstraints(
        maxWidth: 420,
      ),
      padding:
          const EdgeInsets.fromLTRB(
        20,
        18,
        20,
        20,
      ),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin:
              Alignment.topLeft,
          end:
              Alignment.bottomRight,
          colors: [
            const Color(
              0xFF063F42,
            ).withOpacity(0.88),
            const Color(
              0xFF023336,
            ).withOpacity(0.82),
            const Color(
              0xFF021F22,
            ).withOpacity(0.86),
          ],
        ),
        borderRadius:
            BorderRadius.circular(28),
        border: Border.all(
          color:
              AppTheme.tealMint.withOpacity(
            0.24,
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
        ],
      ),
      child: Column(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          _buildProfileHeading(),

          const SizedBox(height: 18),

          _field(
            ctrl: _nameCtrl,
            focusNode: _nameFocus,
            nextFocus:
                _emailFocus,
            hint: 'Full name',
            icon:
                Icons.person_outline_rounded,
            textInputAction:
                TextInputAction.next,
            capitalization:
                TextCapitalization.words,
          ),

          const SizedBox(height: 10),

          _field(
            ctrl: _emailCtrl,
            focusNode: _emailFocus,
            nextFocus:
                _cnicFocus,
            hint: 'Email address',
            icon:
                Icons.mail_outline_rounded,
            type:
                TextInputType.emailAddress,
            textInputAction:
                TextInputAction.next,
          ),

          const SizedBox(height: 10),

          _field(
            ctrl: _cnicCtrl,
            focusNode:
                _cnicFocus,
            nextFocus:
                _cityFocus,
            hint:
                'CNIC (e.g. 3520212345671)',
            icon:
                Icons.badge_outlined,
            type:
                TextInputType.number,
            textInputAction:
                TextInputAction.next,
            formatters: [
              FilteringTextInputFormatter
                  .digitsOnly,
              LengthLimitingTextInputFormatter(
                13,
              ),
            ],
          ),

          const SizedBox(height: 10),

          _field(
            ctrl: _cityCtrl,
            focusNode:
                _cityFocus,
            nextFocus:
                _phoneFocus,
            hint: 'City',
            icon:
                Icons.location_city_outlined,
            textInputAction:
                TextInputAction.next,
            capitalization:
                TextCapitalization.words,
          ),

          const SizedBox(height: 10),

          _field(
            ctrl: _phoneCtrl,
            focusNode:
                _phoneFocus,
            nextFocus:
                _passwordFocus,
            hint:
                'Phone number (03XXXXXXXXX)',
            icon:
                Icons.phone_outlined,
            type:
                TextInputType.phone,
            textInputAction:
                TextInputAction.next,
            formatters: [
              FilteringTextInputFormatter
                  .digitsOnly,
              LengthLimitingTextInputFormatter(
                11,
              ),
            ],
          ),

          const SizedBox(height: 10),

          _field(
            ctrl:
                _passwordCtrl,
            focusNode:
                _passwordFocus,
            nextFocus:
                _confirmPasswordFocus,
            hint:
                'Create password',
            icon:
                Icons.lock_outline_rounded,
            obscure:
                _obscurePassword,
            textInputAction:
                TextInputAction.next,
            suffix:
                IconButton(
              onPressed: () {
                setState(() {
                  _obscurePassword =
                      !_obscurePassword;
                });
              },
              icon: Icon(
                _obscurePassword
                    ? Icons
                        .visibility_outlined
                    : Icons
                        .visibility_off_outlined,
                size: 18,
                color:
                    AppTheme.teal,
              ),
            ),
          ),

          const SizedBox(height: 10),

          _field(
            ctrl:
                _confirmPasswordCtrl,
            focusNode:
                _confirmPasswordFocus,
            hint:
                'Confirm password',
            icon:
                Icons.lock_reset_rounded,
            obscure:
                _obscureConfirmPassword,
            textInputAction:
                TextInputAction.done,
            suffix:
                IconButton(
              onPressed: () {
                setState(() {
                  _obscureConfirmPassword =
                      !_obscureConfirmPassword;
                });
              },
              icon: Icon(
                _obscureConfirmPassword
                    ? Icons
                        .visibility_outlined
                    : Icons
                        .visibility_off_outlined,
                size: 18,
                color:
                    AppTheme.teal,
              ),
            ),
            onSubmitted:
                (_) => _sendOtp(),
          ),

          const SizedBox(height: 11),

          _buildPasswordHint(),

          if (_error != null) ...[
            const SizedBox(height: 12),
            _buildErrorMessage(),
          ],

          const SizedBox(height: 13),

          _buildOtpInformation(),

          const SizedBox(height: 15),

          _buildSendOtpButton(),
        ],
      ),
    );
  }

  // ===========================================================================
  // CARD HEADING
  // ===========================================================================

  Widget _buildProfileHeading() {
    return Row(
      children: [
        Container(
          width: 47,
          height: 47,
          decoration:
              BoxDecoration(
            color: AppTheme.teal
                .withOpacity(
              0.20,
            ),
            borderRadius:
                BorderRadius.circular(
              14,
            ),
            border:
                Border.all(
              color: AppTheme
                  .tealMint
                  .withOpacity(
                0.22,
              ),
            ),
          ),
          child:
              const Icon(
            Icons.person_rounded,
            color:
                AppTheme.tealMint,
            size: 23,
          ),
        ),

        const SizedBox(width: 12),

        Expanded(
          child: Column(
            crossAxisAlignment:
                CrossAxisAlignment.start,
            children: [
              Text(
                'Owner Details',
                style:
                    GoogleFonts.playfairDisplay(
                  color:
                      Colors.white,
                  fontSize: 23,
                  fontWeight:
                      FontWeight.w700,
                ),
              ),

              const SizedBox(height: 3),

              Text(
                'Step 1 of 4 · Personal information',
                style:
                    GoogleFonts.dmSans(
                  color: Colors.white
                      .withOpacity(
                    0.55,
                  ),
                  fontSize: 9.5,
                ),
              ),
            ],
          ),
        ),

        Container(
          padding:
              const EdgeInsets.symmetric(
            horizontal: 9,
            vertical: 6,
          ),
          decoration:
              BoxDecoration(
            color: AppTheme.teal
                .withOpacity(
              0.18,
            ),
            borderRadius:
                BorderRadius.circular(
              14,
            ),
            border:
                Border.all(
              color: AppTheme
                  .tealMint
                  .withOpacity(
                0.20,
              ),
            ),
          ),
          child: Text(
            '1 / 4',
            style:
                GoogleFonts.dmSans(
              color:
                  AppTheme.tealMint,
              fontSize: 9,
              fontWeight:
                  FontWeight.w700,
            ),
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // INPUT FIELD
  // ===========================================================================

  Widget _field({
    required TextEditingController ctrl,
    required FocusNode focusNode,
    required String hint,
    required IconData icon,
    FocusNode? nextFocus,
    TextInputType? type,
    TextInputAction? textInputAction,
    TextCapitalization capitalization =
        TextCapitalization.none,
    List<TextInputFormatter>? formatters,
    ValueChanged<String>? onSubmitted,
    bool obscure = false,
    Widget? suffix,
  }) {
    return AnimatedBuilder(
      animation:
          focusNode,
      builder: (
        context,
        child,
      ) {
        final focused =
            focusNode.hasFocus;

        return AnimatedContainer(
          duration:
              const Duration(
            milliseconds: 180,
          ),
          height: 50,
          decoration:
              BoxDecoration(
            // White boxes as requested.
            color:
                Colors.white.withOpacity(
              0.96,
            ),
            borderRadius:
                BorderRadius.circular(
              15,
            ),
            border:
                Border.all(
              color:
                  focused
                      ? AppTheme.teal
                      : Colors.white
                          .withOpacity(
                          0.60,
                        ),
              width:
                  focused
                      ? 1.4
                      : 0.7,
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black
                    .withOpacity(
                  0.08,
                ),
                blurRadius: 8,
                offset:
                    const Offset(
                  0,
                  3,
                ),
              ),
            ],
          ),
          child: TextField(
            controller: ctrl,
            focusNode: focusNode,
            obscureText: obscure,
            keyboardType: type,
            textInputAction:
                textInputAction,
            textCapitalization:
                capitalization,
            inputFormatters:
                formatters,
            enableSuggestions:
                !obscure,
            autocorrect:
                !obscure,
            onSubmitted:
                onSubmitted ??
                    (_) {
                      if (nextFocus !=
                          null) {
                        FocusScope.of(
                          context,
                        ).requestFocus(
                          nextFocus,
                        );
                      }
                    },

            // Typed input BLACK.
            style:
                GoogleFonts.dmSans(
              fontSize: 13,
              fontWeight:
                  FontWeight.w500,
              color:
                  const Color(
                0xFF102D2E,
              ),
            ),

            cursorColor:
                AppTheme.teal,

            decoration:
                InputDecoration(
              hintText: hint,

              hintStyle:
                  GoogleFonts.dmSans(
                color:
                    const Color(
                  0xFF8A9998,
                ),
                fontSize: 12.5,
              ),

              // Icons remain dark teal.
              prefixIcon:
                  Icon(
                icon,
                size: 19,
                color:
                    AppTheme.teal,
              ),

              suffixIcon:
                  suffix,

              border:
                  InputBorder.none,

              contentPadding:
                  const EdgeInsets
                      .symmetric(
                horizontal: 14,
                vertical: 14,
              ),
            ),
          ),
        );
      },
    );
  }

  // ===========================================================================
  // PASSWORD HINT
  // ===========================================================================

  Widget _buildPasswordHint() {
    return Container(
      width: double.infinity,
      padding:
          const EdgeInsets.symmetric(
        horizontal: 11,
        vertical: 9,
      ),
      decoration:
          BoxDecoration(
        color:
            const Color(
          0xFF01272A,
        ).withOpacity(
          0.58,
        ),
        borderRadius:
            BorderRadius.circular(
          13,
        ),
        border:
            Border.all(
          color: AppTheme
              .tealMint
              .withOpacity(
            0.20,
          ),
        ),
      ),
      child: Row(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.shield_outlined,
            size: 16,
            color:
                AppTheme.tealMint,
          ),

          const SizedBox(width: 8),

          Expanded(
            child: Text(
              'Use at least 8 characters with uppercase, lowercase and a number.',
              style:
                  GoogleFonts.dmSans(
                color: Colors.white
                    .withOpacity(
                  0.62,
                ),
                fontSize: 9.5,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // OTP INFORMATION
  // ===========================================================================

  Widget _buildOtpInformation() {
    return Container(
      width: double.infinity,
      padding:
          const EdgeInsets.all(
        11,
      ),
      decoration:
          BoxDecoration(
        color: AppTheme.teal
            .withOpacity(
          0.18,
        ),
        borderRadius:
            BorderRadius.circular(
          13,
        ),
        border:
            Border.all(
          color: AppTheme
              .tealMint
              .withOpacity(
            0.24,
          ),
        ),
      ),
      child: Row(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          Container(
            width: 31,
            height: 31,
            decoration:
                BoxDecoration(
              color:
                  const Color(
                0xFF0C6D70,
              ).withOpacity(
                0.40,
              ),
              borderRadius:
                  BorderRadius.circular(
                10,
              ),
            ),
            child:
                const Icon(
              Icons.mark_email_read_outlined,
              color:
                  AppTheme.tealMint,
              size: 17,
            ),
          ),

          const SizedBox(width: 9),

          Expanded(
            child: Padding(
              padding:
                  const EdgeInsets.only(
                top: 1,
              ),
              child: Text(
                'A 4-digit OTP will verify your email before your owner account is created.',
                style:
                    GoogleFonts.dmSans(
                  color: Colors.white
                      .withOpacity(
                    0.70,
                  ),
                  fontSize: 9.7,
                  height: 1.4,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // ERROR
  // ===========================================================================

  Widget _buildErrorMessage() {
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
        color:
            const Color(
          0xFF7F2525,
        ).withOpacity(
          0.46,
        ),
        borderRadius:
            BorderRadius.circular(
          12,
        ),
        border:
            Border.all(
          color:
              const Color(
            0xFFFF7777,
          ).withOpacity(
            0.40,
          ),
        ),
      ),
      child: Row(
        crossAxisAlignment:
            CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.error_outline_rounded,
            color:
                Color(
              0xFFFF9696,
            ),
            size: 16,
          ),

          const SizedBox(width: 8),

          Expanded(
            child: Text(
              _error!,
              style:
                  GoogleFonts.dmSans(
                color:
                    const Color(
                  0xFFFFC4C4,
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

  // ===========================================================================
  // SEND OTP
  // ===========================================================================

  Widget _buildSendOtpButton() {
    return SizedBox(
      width:
          double.infinity,
      height: 50,
      child:
          ElevatedButton(
        onPressed:
            _sendOtp,
        style:
            ElevatedButton.styleFrom(
          backgroundColor:
              AppTheme.teal,
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
        child: Row(
          mainAxisAlignment:
              MainAxisAlignment.center,
          children: [
            Text(
              'Continue to Verification',
              style:
                  GoogleFonts.dmSans(
                fontSize: 14,
                fontWeight:
                    FontWeight.w800,
              ),
            ),

            const SizedBox(width: 9),

            const Icon(
              Icons
                  .arrow_forward_rounded,
              size: 18,
            ),
          ],
        ),
      ),
    );
  }
}

// =============================================================================
// STEP INDICATOR
// =============================================================================

class _StepIndicator
    extends StatelessWidget {
  final int step;
  final int total;

  const _StepIndicator({
    required this.step,
    required this.total,
  });

  @override
  Widget build(
    BuildContext context,
  ) {
    return Row(
      children:
          List.generate(
        total,
        (index) {
          final active =
              index < step;

          return Expanded(
            child:
                AnimatedContainer(
              duration:
                  const Duration(
                milliseconds:
                    220,
              ),
              height:
                  active
                      ? 4
                      : 3,
              margin:
                  EdgeInsets.only(
                right:
                    index <
                            total - 1
                        ? 6
                        : 0,
              ),
              decoration:
                  BoxDecoration(
                color:
                    active
                        ? AppTheme
                            .tealMint
                        : Colors
                            .white
                            .withOpacity(
                            0.24,
                          ),
                borderRadius:
                    BorderRadius
                        .circular(
                  3,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}