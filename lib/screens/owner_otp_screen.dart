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
  final String email, phone, name, cnic, city;

  const OwnerOtpScreen({
    super.key,
    required this.email,
    required this.phone,
    required this.name,
    required this.cnic,
    required this.city,
  });

  @override
  State<OwnerOtpScreen> createState() => _OwnerOtpScreenState();
}

class _OwnerOtpScreenState extends State<OwnerOtpScreen>
    with TickerProviderStateMixin {
  final List<TextEditingController> _ctrls =
      List.generate(4, (_) => TextEditingController());
  final List<FocusNode> _foci = List.generate(4, (_) => FocusNode());
  String? _error;
  bool _verifying = false;

  int _secondsLeft = 60;
  Timer? _countdown;

  late final AnimationController _shakeAnim;
  late final Animation<double> _shake;
  late final AnimationController _enterAnim;
  late final Animation<double> _enterSlide;
  late final Animation<double> _enterFade;

  // Demo OTP — in production this would come from email
  static const _demoOtp = '1234';

  @override
  void initState() {
    super.initState();
    _shakeAnim = AnimationController(vsync: this, duration: const Duration(milliseconds: 500));
    _shake = Tween<double>(begin: 0, end: 1).animate(
        CurvedAnimation(parent: _shakeAnim, curve: Curves.elasticOut));

    _enterAnim = AnimationController(vsync: this, duration: const Duration(milliseconds: 800));
    _enterSlide = Tween<double>(begin: 60, end: 0).animate(CurvedAnimation(
        parent: _enterAnim,
        curve: const Interval(0.2, 1.0, curve: Curves.easeOutCubic)));
    _enterFade = CurvedAnimation(parent: _enterAnim,
        curve: const Interval(0.2, 1.0, curve: Curves.easeOut));
    _enterAnim.forward();
    _startCountdown();
    WidgetsBinding.instance.addPostFrameCallback((_) => _foci[0].requestFocus());
  }

  void _startCountdown() {
    _countdown?.cancel();
    setState(() => _secondsLeft = 60);
    _countdown = Timer.periodic(const Duration(seconds: 1), (t) {
      if (!mounted) { t.cancel(); return; }
      setState(() => _secondsLeft--);
      if (_secondsLeft <= 0) t.cancel();
    });
  }

  @override
  void dispose() {
    _countdown?.cancel();
    _shakeAnim.dispose();
    _enterAnim.dispose();
    for (final c in _ctrls) c.dispose();
    for (final f in _foci) f.dispose();
    super.dispose();
  }

  String get _otp => _ctrls.map((c) => c.text).join();

  Future<void> _verify() async {
    if (_otp.length < 4) {
      _triggerError('Please enter all 4 digits');
      return;
    }

    // Accept demo OTP 1234 or any 4 digits (since no real email service)
    // In production: verify against backend
    setState(() { _verifying = true; _error = null; });

    try {
      // Register owner in backend
      await Api().register(
        widget.name,
        widget.email,
        'Owner@${widget.cnic.substring(0, 5)}', // temp password from CNIC
        role: 'owner',
      );
    } catch (e) {
      // Owner might already exist — that's ok, continue
    }

    if (!mounted) return;
    setState(() => _verifying = false);

    Navigator.pushReplacement(context, MaterialPageRoute(
      builder: (_) => OwnerRegisteredScreen(
        name: widget.name,
        email: widget.email,
      )));
  }

  void _triggerError(String msg) {
    setState(() => _error = msg);
    _shakeAnim.forward(from: 0);
  }

  void _resend() {
    if (_secondsLeft > 0) return;
    for (final c in _ctrls) c.clear();
    _foci[0].requestFocus();
    _startCountdown();
    setState(() => _error = null);
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text('Demo OTP: 1234', style: GoogleFonts.dmSans(fontSize: 13)),
      backgroundColor: AppTheme.teal,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      margin: const EdgeInsets.all(16),
    ));
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
                  child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    GestureDetector(
                      onTap: () => Navigator.pop(context),
                      child: Container(width: 38, height: 38,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.15), shape: BoxShape.circle,
                          border: Border.all(color: Colors.white.withOpacity(0.3))),
                        child: const Icon(Icons.arrow_back_rounded, color: Colors.white, size: 18)),
                    ),
                    const SizedBox(height: 16),
                    _StepIndicator(step: 2, total: 4),
                    const SizedBox(height: 14),
                    Text('Verify Email', style: GoogleFonts.playfairDisplay(
                      fontSize: 32, fontWeight: FontWeight.w700, color: Colors.white)),
                    const SizedBox(height: 4),
                    Text('Check your inbox', style: GoogleFonts.dmSans(
                      color: Colors.white.withOpacity(0.6), fontSize: 13)),
                  ]),
                ),
                const Spacer(),
                AnimatedBuilder(
                  animation: _enterAnim,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, _enterSlide.value),
                    child: Opacity(opacity: _enterFade.value, child: child)),
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
      padding: const EdgeInsets.fromLTRB(24, 22, 24, 32),
      decoration: const BoxDecoration(
        color: AppTheme.bg,
        borderRadius: BorderRadius.vertical(top: Radius.circular(32))),
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        Container(width: 40, height: 4, margin: const EdgeInsets.only(bottom: 22),
          decoration: BoxDecoration(color: AppTheme.border, borderRadius: BorderRadius.circular(2))),

        Container(width: 64, height: 64,
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.08), shape: BoxShape.circle,
            border: Border.all(color: AppTheme.teal.withOpacity(0.2))),
          child: Icon(Icons.mark_email_unread_rounded, size: 30, color: AppTheme.teal)),
        const SizedBox(height: 14),

        Text('Enter OTP Code', style: GoogleFonts.playfairDisplay(
          fontSize: 22, fontWeight: FontWeight.w700, color: AppTheme.textPrimary)),
        const SizedBox(height: 8),

        RichText(textAlign: TextAlign.center, text: TextSpan(
          style: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textMuted, height: 1.5),
          children: [
            const TextSpan(text: 'A 4-digit code was sent to\n'),
            TextSpan(text: widget.email, style: GoogleFonts.dmSans(
              fontSize: 13, fontWeight: FontWeight.w600, color: AppTheme.textPrimary)),
          ],
        )),

        // Demo hint
        const SizedBox(height: 10),
        Container(
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.06),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(color: AppTheme.teal.withOpacity(0.2))),
          child: Row(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.info_outline_rounded, color: AppTheme.teal, size: 14),
            const SizedBox(width: 8),
            Text('Demo OTP: 1234', style: GoogleFonts.dmSans(
              fontSize: 12, color: AppTheme.teal, fontWeight: FontWeight.w600)),
          ]),
        ),
        const SizedBox(height: 22),

        // OTP boxes
        AnimatedBuilder(
          animation: _shakeAnim,
          builder: (_, child) {
            final offset = _error != null
                ? 8 * (0.5 - (_shake.value % 1)).abs() * (_shake.value < 0.5 ? 1 : -1)
                : 0.0;
            return Transform.translate(offset: Offset(offset, 0), child: child);
          },
          child: Row(mainAxisAlignment: MainAxisAlignment.center,
            children: List.generate(4, (i) {
              final filled = _ctrls[i].text.isNotEmpty;
              return AnimatedContainer(
                duration: const Duration(milliseconds: 200),
                width: 60, height: 64,
                margin: EdgeInsets.only(right: i < 3 ? 12 : 0),
                decoration: BoxDecoration(
                  color: filled ? AppTheme.teal.withOpacity(0.06) : AppTheme.card,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(
                    color: _error != null ? Colors.red.shade300 : filled ? AppTheme.teal : AppTheme.border,
                    width: filled ? 1.5 : 0.5)),
                child: TextField(
                  controller: _ctrls[i], focusNode: _foci[i],
                  textAlign: TextAlign.center, keyboardType: TextInputType.number,
                  maxLength: 1,
                  inputFormatters: [FilteringTextInputFormatter.digitsOnly],
                  style: GoogleFonts.dmSans(fontSize: 24, fontWeight: FontWeight.w700, color: AppTheme.teal),
                  decoration: const InputDecoration(counterText: '', border: InputBorder.none),
                  onChanged: (val) {
                    setState(() => _error = null);
                    if (val.isNotEmpty && i < 3) _foci[i + 1].requestFocus();
                    if (val.isEmpty && i > 0) _foci[i - 1].requestFocus();
                    if (i == 3 && val.isNotEmpty) _verify();
                    setState(() {});
                  },
                ),
              );
            }),
          ),
        ),
        const SizedBox(height: 16),

        if (_error != null)
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(color: Colors.red.shade50,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.red.shade200, width: 0.5)),
            child: Row(children: [
              Icon(Icons.error_outline_rounded, color: Colors.red.shade600, size: 16),
              const SizedBox(width: 8),
              Expanded(child: Text(_error!, style: GoogleFonts.dmSans(
                color: Colors.red.shade700, fontSize: 12))),
            ]),
          ),

        const SizedBox(height: 16),

        Row(mainAxisAlignment: MainAxisAlignment.center, children: [
          Text("Didn't get the code? ", style: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13)),
          GestureDetector(
            onTap: _secondsLeft <= 0 ? _resend : null,
            child: Text(
              _secondsLeft > 0 ? 'Resend in ${_secondsLeft}s' : 'Resend OTP',
              style: GoogleFonts.dmSans(
                color: _secondsLeft <= 0 ? AppTheme.teal : AppTheme.textMuted,
                fontSize: 13, fontWeight: FontWeight.w700))),
        ]),
        const SizedBox(height: 24),

        SizedBox(width: double.infinity, height: 52,
          child: ElevatedButton(
            onPressed: _verifying ? null : _verify,
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.teal, foregroundColor: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
            child: _verifying
                ? const SizedBox(width: 20, height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white))
                : Text('Verify & Continue', style: GoogleFonts.dmSans(
                    fontSize: 16, fontWeight: FontWeight.w600)),
          )),
      ]),
    );
  }
}

class _StepIndicator extends StatelessWidget {
  final int step, total;
  const _StepIndicator({required this.step, required this.total});

  @override
  Widget build(BuildContext context) {
    return Row(
      children: List.generate(total, (i) => Expanded(
        child: Container(
          height: i == step - 1 ? 4 : 3,
          margin: EdgeInsets.only(right: i < total - 1 ? 5 : 0),
          decoration: BoxDecoration(
            color: i < step ? Colors.white : Colors.white.withOpacity(0.25),
            borderRadius: BorderRadius.circular(2))))),
    );
  }
}
