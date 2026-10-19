import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_registered_screen.dart';

class OwnerOtpScreen extends StatefulWidget {
  final String email;
  final String phone;
  final String name;
  final String cnic;
  final String city;
  final String password;

  const OwnerOtpScreen({
    super.key,
    required this.email,
    required this.phone,
    required this.name,
    required this.cnic,
    required this.city,
    required this.password,
  });

  @override
  State<OwnerOtpScreen> createState() =>
      _OwnerOtpScreenState();
}

class _OwnerOtpScreenState extends State<OwnerOtpScreen>
    with TickerProviderStateMixin {
  final List<TextEditingController> _ctrls =
      List.generate(
    4,
    (_) => TextEditingController(),
  );

  final List<FocusNode> _foci =
      List.generate(
    4,
    (_) => FocusNode(),
  );

  String? _error;

  bool _verifying = false;
  bool _resending = false;

  int _secondsLeft = 60;

  Timer? _countdown;

  late final AnimationController _shakeAnim;
  late final Animation<double> _shake;

  late final AnimationController _enterAnim;
  late final Animation<double> _enterSlide;
  late final Animation<double> _enterFade;

  // ===========================================================================
  // INIT
  // ===========================================================================

  @override
  void initState() {
    super.initState();

    _shakeAnim = AnimationController(
      vsync: this,
      duration: const Duration(
        milliseconds: 500,
      ),
    );

    _shake = Tween<double>(
      begin: 0,
      end: 1,
    ).animate(
      CurvedAnimation(
        parent: _shakeAnim,
        curve: Curves.elasticOut,
      ),
    );

    _enterAnim = AnimationController(
      vsync: this,
      duration: const Duration(
        milliseconds: 750,
      ),
    );

    _enterSlide = Tween<double>(
      begin: 32,
      end: 0,
    ).animate(
      CurvedAnimation(
        parent: _enterAnim,
        curve: Curves.easeOutCubic,
      ),
    );

    _enterFade = CurvedAnimation(
      parent: _enterAnim,
      curve: Curves.easeOut,
    );

    _enterAnim.forward();

    _startCountdown();

    WidgetsBinding.instance.addPostFrameCallback(
      (_) {
        if (mounted) {
          _foci[0].requestFocus();
        }
      },
    );
  }

  // ===========================================================================
  // DISPOSE
  // ===========================================================================

  @override
  void dispose() {
    _countdown?.cancel();

    _shakeAnim.dispose();
    _enterAnim.dispose();

    for (final controller in _ctrls) {
      controller.dispose();
    }

    for (final focusNode in _foci) {
      focusNode.dispose();
    }

    super.dispose();
  }

  // ===========================================================================
  // OTP VALUE
  // ===========================================================================

  String get _otp =>
      _ctrls
          .map(
            (controller) =>
                controller.text,
          )
          .join();

  // ===========================================================================
  // COUNTDOWN
  // ===========================================================================

  void _startCountdown() {
    _countdown?.cancel();

    if (mounted) {
      setState(() {
        _secondsLeft = 60;
      });
    } else {
      _secondsLeft = 60;
    }

    _countdown = Timer.periodic(
      const Duration(seconds: 1),
      (timer) {
        if (!mounted) {
          timer.cancel();
          return;
        }

        if (_secondsLeft <= 1) {
          setState(() {
            _secondsLeft = 0;
          });

          timer.cancel();
          return;
        }

        setState(() {
          _secondsLeft--;
        });
      },
    );
  }

  // ===========================================================================
  // VERIFY OTP
  // ===========================================================================

  Future<void> _verify() async {
    if (_verifying) return;

    FocusScope.of(context).unfocus();

    final otp = _otp.trim();

    if (otp.length != 4) {
      _triggerError(
        'Please enter all 4 digits.',
      );
      return;
    }

    setState(() {
      _verifying = true;
      _error = null;
    });

    try {
      final result =
          await Api().verifyOwnerOtp(
        email: widget.email,
        otp: otp,
        password: widget.password,
      );

      final token =
          result['token']?.toString();

      final userRaw =
          result['user'];

      if (token == null ||
          token.isEmpty ||
          userRaw is! Map) {
        throw Exception(
          'Owner verification succeeded but login information was not returned.',
        );
      }

      final user =
          Map<String, dynamic>.from(
        userRaw,
      );

      await AuthStore.saveToken(
        token,
      );

      await AuthStore.saveUser(
        user,
      );

      if (!mounted) return;

      setState(() {
        _verifying = false;
      });

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) =>
              OwnerRegisteredScreen(
            name: widget.name,
            email: widget.email,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _verifying = false;
      });

      final message = e
          .toString()
          .replaceFirst(
            'Exception: ',
            '',
          )
          .trim();

      _triggerError(
        message.isEmpty
            ? 'Unable to verify OTP.'
            : message,
      );
    }
  }

  // ===========================================================================
  // ERROR
  // ===========================================================================

  void _triggerError(
    String message,
  ) {
    if (!mounted) return;

    setState(() {
      _error = message;
    });

    _shakeAnim.forward(
      from: 0,
    );
  }

  // ===========================================================================
  // RESEND OTP
  // ===========================================================================

  Future<void> _resend() async {
    if (_secondsLeft > 0 ||
        _resending) {
      return;
    }

    setState(() {
      _resending = true;
      _error = null;
    });

    try {
      await Api().sendOwnerOtp(
        name: widget.name,
        email: widget.email,
        password: widget.password,
        cnic: widget.cnic,
        city: widget.city,
        phone: widget.phone,
      );

      for (final controller in _ctrls) {
        controller.clear();
      }

      if (!mounted) return;

      _foci[0].requestFocus();

      _startCountdown();

      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(
          SnackBar(
            behavior:
                SnackBarBehavior.floating,
            margin:
                const EdgeInsets.all(16),
            backgroundColor:
                const Color(
              0xFF063F42,
            ),
            shape:
                RoundedRectangleBorder(
              borderRadius:
                  BorderRadius.circular(
                14,
              ),
            ),
            content: Row(
              children: [
                const Icon(
                  Icons
                      .mark_email_read_rounded,
                  color:
                      AppTheme.tealMint,
                  size: 18,
                ),
                const SizedBox(
                  width: 10,
                ),
                Expanded(
                  child: Text(
                    'A new OTP has been sent to ${widget.email}',
                    style:
                        GoogleFonts.dmSans(
                      color: Colors.white,
                      fontSize: 12,
                      fontWeight:
                          FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
    } catch (e) {
      if (!mounted) return;

      _triggerError(
        e
            .toString()
            .replaceFirst(
              'Exception: ',
              '',
            ),
      );
    } finally {
      if (mounted) {
        setState(() {
          _resending = false;
        });
      }
    }
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    final keyboardHeight =
        MediaQuery.viewInsetsOf(
      context,
    ).bottom;

    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset: true,
        backgroundColor:
            Colors.transparent,
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
                    constraints:
                        BoxConstraints(
                      minHeight:
                          constraints
                                  .maxHeight -
                              36,
                    ),
                    child: Column(
                      crossAxisAlignment:
                          CrossAxisAlignment
                              .start,
                      children: [
                        _buildHeader(),

                        const SizedBox(
                          height: 20,
                        ),

                        AnimatedBuilder(
                          animation:
                              _enterAnim,
                          builder: (
                            context,
                            child,
                          ) {
                            return Transform
                                .translate(
                              offset: Offset(
                                0,
                                _enterSlide
                                    .value,
                              ),
                              child: Opacity(
                                opacity:
                                    _enterFade
                                        .value,
                                child:
                                    child,
                              ),
                            );
                          },
                          child:
                              _buildOtpCard(),
                        ),

                        const SizedBox(
                          height: 24,
                        ),
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
                Navigator.maybePop(
              context,
            ),
            borderRadius:
                BorderRadius.circular(
              50,
            ),
            child: Container(
              width: 42,
              height: 42,
              decoration:
                  BoxDecoration(
                color:
                    const Color(
                  0xFF02383B,
                ).withOpacity(
                  0.48,
                ),
                shape:
                    BoxShape.circle,
                border:
                    Border.all(
                  color: Colors.white
                      .withOpacity(
                    0.25,
                  ),
                ),
              ),
              child:
                  const Icon(
                Icons
                    .arrow_back_rounded,
                color:
                    Colors.white,
                size: 20,
              ),
            ),
          ),

          const SizedBox(
            height: 16,
          ),

          const _StepIndicator(
            step: 2,
            total: 4,
          ),

          const SizedBox(
            height: 18,
          ),

          Text(
            'Verify Your\nEmail',
            style:
                GoogleFonts
                    .playfairDisplay(
              fontSize: 37,
              height: 1.05,
              fontWeight:
                  FontWeight.w700,
              color:
                  Colors.white,
            ),
          ),

          const SizedBox(
            height: 12,
          ),

          Text(
            'Enter the verification code we sent to your email.',
            style:
                GoogleFonts.dmSans(
              color: Colors.white
                  .withOpacity(
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
  // DARK TEAL GLASS OTP CARD
  // ===========================================================================

  Widget _buildOtpCard() {
    return Center(
      child: Container(
        width: double.infinity,
        constraints:
            const BoxConstraints(
          maxWidth: 420,
        ),
        padding:
            const EdgeInsets.fromLTRB(
          20,
          20,
          20,
          21,
        ),
        decoration:
            BoxDecoration(
          gradient:
              LinearGradient(
            begin:
                Alignment.topLeft,
            end:
                Alignment.bottomRight,
            colors: [
              const Color(
                0xFF063F42,
              ).withOpacity(
                0.88,
              ),
              const Color(
                0xFF023336,
              ).withOpacity(
                0.82,
              ),
              const Color(
                0xFF021F22,
              ).withOpacity(
                0.86,
              ),
            ],
          ),
          borderRadius:
              BorderRadius.circular(
            28,
          ),
          border:
              Border.all(
            color: AppTheme
                .tealMint
                .withOpacity(
              0.24,
            ),
            width: 0.9,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black
                  .withOpacity(
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
          mainAxisSize:
              MainAxisSize.min,
          children: [
            // ===============================================================
            // EMAIL ICON
            // ===============================================================

            Container(
              width: 58,
              height: 58,
              decoration:
                  BoxDecoration(
                color: AppTheme.teal
                    .withOpacity(
                  0.22,
                ),
                shape:
                    BoxShape.circle,
                border:
                    Border.all(
                  color: AppTheme
                      .tealMint
                      .withOpacity(
                    0.28,
                  ),
                ),
              ),
              child:
                  const Icon(
                Icons
                    .mark_email_unread_rounded,
                size: 27,
                color:
                    AppTheme.tealMint,
              ),
            ),

            const SizedBox(
              height: 13,
            ),

            Text(
              'Enter OTP Code',
              textAlign:
                  TextAlign.center,
              style:
                  GoogleFonts
                      .playfairDisplay(
                fontSize: 23,
                fontWeight:
                    FontWeight.w700,
                color:
                    Colors.white,
              ),
            ),

            const SizedBox(
              height: 7,
            ),

            Text(
              'We sent a 4-digit verification code to',
              textAlign:
                  TextAlign.center,
              style:
                  GoogleFonts.dmSans(
                fontSize: 10.5,
                color: Colors.white
                    .withOpacity(
                  0.57,
                ),
              ),
            ),

            const SizedBox(
              height: 4,
            ),

            // ===============================================================
            // EMAIL PILL
            // ===============================================================

            Container(
              constraints:
                  const BoxConstraints(
                maxWidth: 330,
              ),
              padding:
                  const EdgeInsets
                      .symmetric(
                horizontal: 11,
                vertical: 7,
              ),
              decoration:
                  BoxDecoration(
                color:
                    const Color(
                  0xFF01272A,
                ).withOpacity(
                  0.60,
                ),
                borderRadius:
                    BorderRadius.circular(
                  12,
                ),
                border:
                    Border.all(
                  color: AppTheme
                      .tealMint
                      .withOpacity(
                    0.16,
                  ),
                ),
              ),
              child: Row(
                mainAxisSize:
                    MainAxisSize.min,
                children: [
                  const Icon(
                    Icons
                        .mail_outline_rounded,
                    color: AppTheme
                        .tealMint,
                    size: 14,
                  ),

                  const SizedBox(
                    width: 7,
                  ),

                  Flexible(
                    child: Text(
                      widget.email,
                      overflow:
                          TextOverflow
                              .ellipsis,
                      style:
                          GoogleFonts
                              .dmSans(
                        color:
                            Colors.white,
                        fontSize: 10.5,
                        fontWeight:
                            FontWeight
                                .w600,
                      ),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(
              height: 21,
            ),

            // ===============================================================
            // OTP BOXES
            // ===============================================================

            AnimatedBuilder(
              animation:
                  _shakeAnim,
              builder: (
                context,
                child,
              ) {
                double offset = 0;

                if (_error != null) {
                  offset =
                      7 *
                      (0.5 -
                              (_shake.value %
                                  1))
                          .abs() *
                      (_shake.value <
                              0.5
                          ? 1
                          : -1);
                }

                return Transform
                    .translate(
                  offset:
                      Offset(
                    offset,
                    0,
                  ),
                  child:
                      child,
                );
              },
              child:
                  _buildOtpBoxes(),
            ),

            if (_error != null) ...[
              const SizedBox(
                height: 13,
              ),
              _buildErrorMessage(),
            ],

            const SizedBox(
              height: 18,
            ),

            // ===============================================================
            // RESEND
            // ===============================================================

            _buildResendArea(),

            const SizedBox(
              height: 19,
            ),

            // ===============================================================
            // VERIFY BUTTON
            // ===============================================================

            _buildVerifyButton(),

            const SizedBox(
              height: 13,
            ),

            // ===============================================================
            // SECURITY NOTE
            // ===============================================================

            _buildSecurityNote(),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // OTP BOXES
  // ===========================================================================

  Widget _buildOtpBoxes() {
    return LayoutBuilder(
      builder: (
        context,
        constraints,
      ) {
        const gap = 9.0;

        final availableWidth =
            constraints.maxWidth -
                (gap * 3);

        final calculated =
            availableWidth / 4;

        final boxWidth =
            calculated
                .clamp(
                  45.0,
                  62.0,
                )
                .toDouble();

        return Row(
          mainAxisAlignment:
              MainAxisAlignment.center,
          children:
              List.generate(
            4,
            (index) {
              final filled =
                  _ctrls[index]
                      .text
                      .isNotEmpty;

              final focused =
                  _foci[index]
                      .hasFocus;

              return Padding(
                padding:
                    EdgeInsets.only(
                  right:
                      index < 3
                          ? gap
                          : 0,
                ),
                child:
                    AnimatedContainer(
                  duration:
                      const Duration(
                    milliseconds:
                        180,
                  ),
                  width:
                      boxWidth,
                  height: 58,
                  decoration:
                      BoxDecoration(
                    // WHITE BOX
                    color: Colors.white
                        .withOpacity(
                      0.97,
                    ),
                    borderRadius:
                        BorderRadius
                            .circular(
                      15,
                    ),
                    border:
                        Border.all(
                      color:
                          _error !=
                                  null
                              ? const Color(
                                  0xFFD95B5B,
                                )
                              : focused ||
                                      filled
                                  ? AppTheme
                                      .teal
                                  : Colors
                                      .white,
                      width:
                          focused
                              ? 1.8
                              : filled
                                  ? 1.3
                                  : 0.7,
                    ),
                    boxShadow:
                        focused
                            ? [
                                BoxShadow(
                                  color: AppTheme
                                      .tealMint
                                      .withOpacity(
                                    0.18,
                                  ),
                                  blurRadius:
                                      12,
                                ),
                              ]
                            : [
                                BoxShadow(
                                  color: Colors
                                      .black
                                      .withOpacity(
                                    0.08,
                                  ),
                                  blurRadius:
                                      7,
                                  offset:
                                      const Offset(
                                    0,
                                    3,
                                  ),
                                ),
                              ],
                  ),
                  child:
                      TextField(
                    controller:
                        _ctrls[index],
                    focusNode:
                        _foci[index],
                    textAlign:
                        TextAlign
                            .center,
                    keyboardType:
                        TextInputType
                            .number,
                    textInputAction:
                        index < 3
                            ? TextInputAction
                                .next
                            : TextInputAction
                                .done,
                    maxLength: 1,
                    inputFormatters: [
                      FilteringTextInputFormatter
                          .digitsOnly,
                    ],

                    // DARK TEAL OTP DIGITS
                    style:
                        GoogleFonts
                            .dmSans(
                      fontSize: 22,
                      fontWeight:
                          FontWeight
                              .w800,
                      color:
                          const Color(
                        0xFF064B4E,
                      ),
                    ),

                    cursorColor:
                        AppTheme.teal,

                    decoration:
                        const InputDecoration(
                      counterText:
                          '',
                      border:
                          InputBorder
                              .none,
                      contentPadding:
                          EdgeInsets
                              .zero,
                    ),

                    onTap: () {
                      setState(
                        () {},
                      );
                    },

                    onChanged:
                        (value) {
                      setState(() {
                        _error =
                            null;
                      });

                      if (value
                              .isNotEmpty &&
                          index < 3) {
                        _foci[
                                index +
                                    1]
                            .requestFocus();
                      }

                      if (value
                              .isEmpty &&
                          index > 0) {
                        _foci[
                                index -
                                    1]
                            .requestFocus();
                      }

                      if (index ==
                              3 &&
                          value
                              .isNotEmpty &&
                          _otp.length ==
                              4) {
                        _verify();
                      }
                    },
                  ),
                ),
              );
            },
          ),
        );
      },
    );
  }

  // ===========================================================================
  // RESEND AREA
  // ===========================================================================

  Widget _buildResendArea() {
    final canResend =
        _secondsLeft <= 0 &&
            !_resending;

    return Wrap(
      alignment:
          WrapAlignment.center,
      crossAxisAlignment:
          WrapCrossAlignment.center,
      children: [
        Text(
          "Didn't receive the code? ",
          style:
              GoogleFonts.dmSans(
            color: Colors.white
                .withOpacity(
              0.55,
            ),
            fontSize: 11.5,
          ),
        ),

        InkWell(
          onTap:
              canResend
                  ? _resend
                  : null,
          borderRadius:
              BorderRadius.circular(
            8,
          ),
          child: Padding(
            padding:
                const EdgeInsets
                    .symmetric(
              horizontal: 2,
              vertical: 4,
            ),
            child: Text(
              _resending
                  ? 'Sending...'
                  : _secondsLeft > 0
                      ? 'Resend in ${_secondsLeft}s'
                      : 'Resend OTP',
              style:
                  GoogleFonts.dmSans(
                color:
                    canResend
                        ? AppTheme
                            .tealMint
                        : Colors.white
                            .withOpacity(
                            0.42,
                          ),
                fontSize: 11.5,
                fontWeight:
                    FontWeight.w700,
              ),
            ),
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // VERIFY BUTTON
  // ===========================================================================

  Widget _buildVerifyButton() {
    return SizedBox(
      width:
          double.infinity,
      height: 50,
      child:
          ElevatedButton(
        onPressed:
            _verifying
                ? null
                : _verify,
        style:
            ElevatedButton.styleFrom(
          backgroundColor:
              AppTheme.teal,
          disabledBackgroundColor:
              AppTheme.teal
                  .withOpacity(
            0.55,
          ),
          foregroundColor:
              Colors.white,
          disabledForegroundColor:
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
            _verifying
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child:
                        CircularProgressIndicator(
                      strokeWidth:
                          2,
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
                            .verified_rounded,
                        size: 18,
                      ),

                      const SizedBox(
                        width: 8,
                      ),

                      Text(
                        'Verify & Continue',
                        style:
                            GoogleFonts
                                .dmSans(
                          fontSize:
                              14,
                          fontWeight:
                              FontWeight
                                  .w800,
                        ),
                      ),
                    ],
                  ),
      ),
    );
  }

  // ===========================================================================
  // SECURITY NOTE
  // ===========================================================================

  Widget _buildSecurityNote() {
    return Container(
      width:
          double.infinity,
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
          0.55,
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
            0.18,
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
                0.22,
              ),
              borderRadius:
                  BorderRadius.circular(
                9,
              ),
            ),
            child:
                const Icon(
              Icons
                  .shield_outlined,
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
              'For your security, this code can only be used to verify this account.',
              style:
                  GoogleFonts.dmSans(
                color: Colors.white
                    .withOpacity(
                  0.58,
                ),
                fontSize: 9.5,
                height: 1.35,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // ERROR MESSAGE
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
          0.45,
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
            Icons
                .error_outline_rounded,
            color:
                Color(
              0xFFFF9696,
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