import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_otp_screen.dart';

class OwnerInfoScreen extends StatefulWidget {
  const OwnerInfoScreen({super.key});

  @override
  State<OwnerInfoScreen> createState() => _OwnerInfoScreenState();
}

class _OwnerInfoScreenState extends State<OwnerInfoScreen>
    with SingleTickerProviderStateMixin {
  final _nameCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _cnicCtrl = TextEditingController();
  final _cityCtrl = TextEditingController();
  final _phoneCtrl = TextEditingController();
  String? _error;

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(vsync: this, duration: const Duration(milliseconds: 900));
    _fade = CurvedAnimation(parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOut));
    _slide = Tween<double>(begin: 70, end: 0).animate(CurvedAnimation(
        parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOutCubic)));
    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    for (final c in [_nameCtrl, _emailCtrl, _cnicCtrl, _cityCtrl, _phoneCtrl]) {
      c.dispose();
    }
    super.dispose();
  }

  void _sendOtp() {
    final name = _nameCtrl.text.trim();
    final email = _emailCtrl.text.trim();
    final cnic = _cnicCtrl.text.trim();
    final city = _cityCtrl.text.trim();
    final phone = _phoneCtrl.text.trim();

    if ([name, email, cnic, city, phone].any((s) => s.isEmpty)) {
      setState(() => _error = 'Please fill all fields');
      return;
    }

    setState(() => _error = null);

    Navigator.pushReplacement(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerOtpScreen(
          email: email,
          phone: phone,
          name: name,
          cnic: cnic,
          city: city,
        ),
      ),
    );
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
                            border: Border.all(color: Colors.white.withOpacity(0.3)),
                          ),
                          child: const Icon(Icons.arrow_back_rounded,
                              color: Colors.white, size: 18),
                        ),
                      ),
                      const SizedBox(height: 16),
                      _StepIndicator(step: 1, total: 4),
                      const SizedBox(height: 14),
                      Text('Your Profile',
                        style: GoogleFonts.playfairDisplay(
                          fontSize: 32, fontWeight: FontWeight.w700,
                          color: Colors.white)),
                      const SizedBox(height: 4),
                      Text('Tell us about yourself',
                        style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.6), fontSize: 13)),
                    ],
                  ),
                ),
                const Spacer(),
                AnimatedBuilder(
                  animation: _anim,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, _slide.value),
                    child: Opacity(opacity: _fade.value, child: child),
                  ),
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
      padding: const EdgeInsets.fromLTRB(24, 20, 24, 28),
      decoration: const BoxDecoration(
        color: AppTheme.bg,
        borderRadius: BorderRadius.vertical(top: Radius.circular(32)),
      ),
      child: SingleChildScrollView(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Center(
              child: Container(
                width: 40, height: 4,
                margin: const EdgeInsets.only(bottom: 20),
                decoration: BoxDecoration(
                  color: AppTheme.border,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
            ),

            Row(crossAxisAlignment: CrossAxisAlignment.center, children: [
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text('Fill your profile',
                    style: GoogleFonts.playfairDisplay(
                      fontSize: 22, fontWeight: FontWeight.w700,
                      color: AppTheme.textPrimary)),
                  const SizedBox(height: 3),
                  Text('Step 1 of 4',
                    style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.textMuted)),
                ]),
              ),
              Stack(children: [
                Container(
                  width: 62, height: 62,
                  decoration: BoxDecoration(
                    color: AppTheme.teal.withOpacity(0.1),
                    shape: BoxShape.circle,
                    border: Border.all(color: AppTheme.teal.withOpacity(0.2), width: 2),
                  ),
                  child: Icon(Icons.person_rounded, size: 32, color: AppTheme.teal),
                ),
                Positioned(
                  bottom: 0, right: 0,
                  child: Container(
                    width: 20, height: 20,
                    decoration: BoxDecoration(
                      color: AppTheme.teal, shape: BoxShape.circle,
                      border: Border.all(color: Colors.white, width: 2),
                    ),
                    child: const Icon(Icons.edit_rounded, size: 10, color: Colors.white),
                  ),
                ),
              ]),
            ]),
            const SizedBox(height: 20),

            _field(ctrl: _nameCtrl, hint: 'Full name', icon: Icons.person_outline_rounded),
            const SizedBox(height: 11),
            _field(ctrl: _emailCtrl, hint: 'Email address',
                icon: Icons.mail_outline_rounded, type: TextInputType.emailAddress),
            const SizedBox(height: 11),
            _field(ctrl: _cnicCtrl, hint: 'CNIC (e.g. 35202-1234567-1)',
                icon: Icons.badge_outlined, type: TextInputType.number,
                formatters: [
                  FilteringTextInputFormatter.digitsOnly,
                  LengthLimitingTextInputFormatter(13),
                ]),
            const SizedBox(height: 11),
            _field(ctrl: _cityCtrl, hint: 'City', icon: Icons.location_city_outlined),
            const SizedBox(height: 11),
            _field(ctrl: _phoneCtrl, hint: 'Phone number (03XXXXXXXXX)',
                icon: Icons.phone_outlined, type: TextInputType.phone,
                formatters: [
                  FilteringTextInputFormatter.digitsOnly,
                  LengthLimitingTextInputFormatter(11),
                ]),
            const SizedBox(height: 16),

            if (_error != null)
              Container(
                margin: const EdgeInsets.only(bottom: 14),
                padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 11),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.red.shade200, width: 0.5),
                ),
                child: Row(children: [
                  Icon(Icons.error_outline_rounded, color: Colors.red.shade600, size: 16),
                  const SizedBox(width: 8),
                  Expanded(child: Text(_error!,
                    style: GoogleFonts.dmSans(color: Colors.red.shade700, fontSize: 12))),
                ]),
              ),

            Container(
              padding: const EdgeInsets.all(13),
              decoration: BoxDecoration(
                color: AppTheme.teal.withOpacity(0.06),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppTheme.teal.withOpacity(0.18), width: 0.5),
              ),
              child: Row(children: [
                Icon(Icons.mark_email_read_outlined, color: AppTheme.teal, size: 18),
                const SizedBox(width: 10),
                Expanded(child: Text(
                  'A 4-digit OTP will be sent to your email for verification.',
                  style: GoogleFonts.dmSans(
                    color: AppTheme.teal, fontSize: 12, fontWeight: FontWeight.w400))),
              ]),
            ),
            const SizedBox(height: 18),

            SizedBox(
              width: double.infinity, height: 52,
              child: ElevatedButton(
                onPressed: _sendOtp,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.teal,
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16)),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text('Send OTP',
                      style: GoogleFonts.dmSans(
                          fontSize: 16, fontWeight: FontWeight.w600)),
                    const SizedBox(width: 10),
                    const Icon(Icons.send_rounded, size: 18),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _field({
    required TextEditingController ctrl,
    required String hint,
    required IconData icon,
    TextInputType? type,
    List<TextInputFormatter>? formatters,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border, width: 0.5),
      ),
      child: TextField(
        controller: ctrl,
        keyboardType: type,
        inputFormatters: formatters,
        style: GoogleFonts.dmSans(fontSize: 14, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13),
          prefixIcon: Icon(icon, size: 20, color: AppTheme.textMuted),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
        ),
      ),
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
            borderRadius: BorderRadius.circular(2),
          ),
        ),
      )),
    );
  }
}