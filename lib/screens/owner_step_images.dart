import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_hostel_success_screen.dart';
import 'package:image_picker/image_picker.dart';

// ════════════════════════════════════════════════════════════════════════════
// STEP 9 — Photos & Videos
// ════════════════════════════════════════════════════════════════════════════
class OwnerStepImages extends StatefulWidget {
  final String ownerEmail, ownerName, hostelName, hostelType, city, address;
  final double latitude, longitude;

  const OwnerStepImages({
    super.key,
    required this.ownerEmail, required this.ownerName,
    required this.hostelName, required this.hostelType,
    required this.city, required this.address,
    required this.latitude, required this.longitude,
  });

  @override
  State<OwnerStepImages> createState() => _OwnerStepImagesState();
}

class _OwnerStepImagesState extends State<OwnerStepImages>
    with SingleTickerProviderStateMixin {
  bool _hasExterior = false;
  bool _hasRoom     = false;
  bool _hasCommon   = false;
  bool _hasVideo    = false;

  late final AnimationController _anim;
  late final Animation<double> _fade, _slide;

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(vsync: this,
        duration: const Duration(milliseconds: 800));
    _fade  = CurvedAnimation(parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOut));
    _slide = Tween<double>(begin: 60, end: 0).animate(CurvedAnimation(
        parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOutCubic)));
    _anim.forward();
  }

  @override
  void dispose() { _anim.dispose(); super.dispose(); }

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
            child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Row(children: [
                    GestureDetector(
                      onTap: () => Navigator.pop(context),
                      child: Container(
                        width: 38, height: 38,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.15),
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white.withOpacity(0.3))),
                        child: const Icon(Icons.arrow_back_rounded,
                            color: Colors.white, size: 18)),
                    ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: Colors.white.withOpacity(0.2))),
                      child: Text('Step 9 of 9',
                          style: GoogleFonts.dmSans(
                              color: Colors.white, fontSize: 12,
                              fontWeight: FontWeight.w600)),
                    ),
                  ]),
                  const SizedBox(height: 16),
                  Row(children: List.generate(9, (i) => Expanded(
                    child: Container(
                      height: 4,
                      margin: EdgeInsets.only(right: i < 8 ? 4 : 0),
                      decoration: BoxDecoration(
                        color: AppTheme.tealMint,
                        borderRadius: BorderRadius.circular(2)),
                    ),
                  ))),
                  const SizedBox(height: 14),
                  Text('Hostel Photos & Videos',
                      style: GoogleFonts.playfairDisplay(
                          fontSize: 30, fontWeight: FontWeight.w700,
                          color: Colors.white)),
                  const SizedBox(height: 4),
                  Text('Show students what to expect',
                      style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.6), fontSize: 13)),
                ]),
              ),
              Expanded(
                child: AnimatedBuilder(
                  animation: _anim,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, _slide.value),
                    child: Opacity(opacity: _fade.value, child: child),
                  ),
                  child: DraggableScrollableSheet(
                    initialChildSize: 0.85,
                    minChildSize: 0.45,
                    maxChildSize: 0.96,
                    expand: true,
                    builder: (ctx, scrollCtrl) => _buildSheet(scrollCtrl),
                  ),
                ),
              ),
            ]),
          ),
        ),
      ),
    );
  }


  Future<void> _pickImage(ValueChanged<bool> onDone) async {
  final picker = ImagePicker();
  final file = await picker.pickImage(
    source: ImageSource.gallery,
    imageQuality: 80,
  );
  if (file != null) onDone(true);
}

Future<void> _pickVideo(ValueChanged<bool> onDone) async {
  final picker = ImagePicker();
  final file = await picker.pickVideo(
    source: ImageSource.gallery,
  );
  if (file != null) onDone(true);
}

  Widget _buildSheet(ScrollController scrollCtrl) {
    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        color: AppTheme.bg,
        borderRadius: BorderRadius.vertical(top: Radius.circular(32)),
      ),
      child: Column(children: [
        const SizedBox(height: 12),
        Center(child: Container(width: 40, height: 4,
          decoration: BoxDecoration(color: AppTheme.border,
              borderRadius: BorderRadius.circular(2)))),
        Expanded(
          child: SingleChildScrollView(
            controller: scrollCtrl,
            padding: const EdgeInsets.fromLTRB(20, 14, 20, 32),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                children: [
              // Tip
              Container(
                padding: const EdgeInsets.all(13),
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.05),
                  borderRadius: BorderRadius.circular(12),
                  border: Border(left: BorderSide(color: AppTheme.teal, width: 3))),
                child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  const Icon(Icons.photo_camera_rounded, color: AppTheme.teal, size: 16),
                  const SizedBox(width: 10),
                  Expanded(child: Text(
                    'Hostels with photos get 3× more views. Add at least the exterior photo.',
                    style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.teal, height: 1.4))),
                ]),
              ),
              const SizedBox(height: 16),

              _uploadCard(
                icon: Icons.home_rounded, color: AppTheme.teal,
                title: 'Hostel Exterior', subtitle: 'Main building front view',
                uploaded: _hasExterior,
                onTap: () => _pickImage((v) => setState(() => _hasExterior = v)),
              ),
              const SizedBox(height: 10),
              _uploadCard(
                icon: Icons.bed_rounded, color: const Color(0xFF185FA5),
                title: 'Room Photos', subtitle: 'Interior room view',
                uploaded: _hasRoom,
                onTap: () => _pickImage((v) => setState(() => _hasRoom = v)),
              ),
              const SizedBox(height: 10),
              _uploadCard(
                icon: Icons.weekend_rounded, color: const Color(0xFF7B3FC4),
                title: 'Common Areas', subtitle: 'Lounge, dining, corridors',
                uploaded: _hasCommon,
                onTap: () => _pickImage((v) => setState(() => _hasCommon = v)),
              ),
              const SizedBox(height: 10),
              _uploadCard(
                icon: Icons.videocam_rounded, color: const Color(0xFFB85C00),
                title: 'Walkthrough Video', subtitle: 'Optional — max 2 minutes',
                uploaded: _hasVideo, isOptional: true,
                onTap: () => _pickVideo((v) => setState(() => _hasVideo = v)),
              ),
              const SizedBox(height: 20),

              SizedBox(
                width: double.infinity, height: 52,
                child: ElevatedButton(
                  onPressed: () => Navigator.push(context, MaterialPageRoute(
                    builder: (_) => OwnerStepReview(
                      ownerEmail: widget.ownerEmail, ownerName: widget.ownerName,
                      hostelName: widget.hostelName, hostelType: widget.hostelType,
                      city: widget.city, address: widget.address,
                    ),
                  )),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.teal, foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                  child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                    Text('Next  →  Review & Submit',
                        style: GoogleFonts.dmSans(fontSize: 15, fontWeight: FontWeight.w600)),
                    const SizedBox(width: 10),
                    Container(
                      width: 26, height: 26,
                      decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.2), shape: BoxShape.circle),
                      child: const Icon(Icons.fact_check_rounded, size: 14)),
                  ]),
                ),
              ),
            ]),
          ),
        ),
      ]),
    );
  }

  Widget _uploadCard({
    required IconData icon, required Color color,
    required String title, required String subtitle,
    required bool uploaded, required VoidCallback onTap,
    bool isOptional = false,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: uploaded ? color.withOpacity(0.05) : AppTheme.card,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: uploaded ? color : AppTheme.border,
              width: uploaded ? 1.5 : 0.5)),
        child: Row(children: [
          Container(
            width: 48, height: 48,
            decoration: BoxDecoration(
              color: uploaded ? color.withOpacity(0.12) : AppTheme.bgSecondary,
              borderRadius: BorderRadius.circular(12)),
            child: Icon(uploaded ? Icons.check_circle_rounded : icon,
                size: 24, color: uploaded ? color : AppTheme.textMuted),
          ),
          const SizedBox(width: 14),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              Text(title, style: GoogleFonts.dmSans(
                  fontSize: 13, fontWeight: FontWeight.w700,
                  color: uploaded ? color : AppTheme.textPrimary)),
              if (isOptional) ...[
                const SizedBox(width: 6),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                  decoration: BoxDecoration(color: AppTheme.bgSecondary,
                      borderRadius: BorderRadius.circular(6)),
                  child: Text('Optional', style: GoogleFonts.dmSans(
                      fontSize: 10, color: AppTheme.textMuted))),
              ],
            ]),
            Text(subtitle, style: GoogleFonts.dmSans(fontSize: 11, color: AppTheme.textMuted)),
          ])),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
            decoration: BoxDecoration(
              color: uploaded ? color.withOpacity(0.1) : AppTheme.teal.withOpacity(0.08),
              borderRadius: BorderRadius.circular(10)),
            child: Text(uploaded ? 'Done ✓' : 'Upload',
                style: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w600,
                    color: uploaded ? color : AppTheme.teal)),
          ),
        ]),
      ),
    );
  }
}


// ════════════════════════════════════════════════════════════════════════════
// STEP 10 — Review & Submit
// ════════════════════════════════════════════════════════════════════════════
class OwnerStepReview extends StatefulWidget {
  final String ownerEmail, ownerName, hostelName, hostelType, city, address;

  const OwnerStepReview({
    super.key,
    required this.ownerEmail, required this.ownerName,
    required this.hostelName, required this.hostelType,
    required this.city, required this.address,
  });

  @override
  State<OwnerStepReview> createState() => _OwnerStepReviewState();
}

class _OwnerStepReviewState extends State<OwnerStepReview>
    with SingleTickerProviderStateMixin {
  final _summaryCtrl    = TextEditingController();
  final _facilitiesCtrl = TextEditingController();
  final _rulesCtrl      = TextEditingController();
  bool _agreed     = false;
  bool _submitting = false;

  late final AnimationController _anim;
  late final Animation<double> _fade, _slide;

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(vsync: this,
        duration: const Duration(milliseconds: 800));
    _fade  = CurvedAnimation(parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOut));
    _slide = Tween<double>(begin: 60, end: 0).animate(CurvedAnimation(
        parent: _anim,
        curve: const Interval(0.15, 1.0, curve: Curves.easeOutCubic)));
    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    _summaryCtrl.dispose();
    _facilitiesCtrl.dispose();
    _rulesCtrl.dispose();
    super.dispose();
  }

  void _submit() async {
    if (!_agreed) {
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(
        content: const Text('Please agree to the terms first'),
        backgroundColor: Colors.red.shade600,
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
        margin: const EdgeInsets.all(16)));
      return;
    }
    setState(() => _submitting = true);
    await Future.delayed(const Duration(seconds: 2));
    if (!mounted) return;
    Navigator.pushReplacement(context, MaterialPageRoute(
      builder: (_) => OwnerHostelSuccessScreen(
        ownerName: widget.ownerName,
        hostelName: widget.hostelName,
      ),
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
            child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                children: [
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 14, 20, 0),
                child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Row(children: [
                    GestureDetector(
                      onTap: () => Navigator.pop(context),
                      child: Container(
                        width: 38, height: 38,
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.15),
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white.withOpacity(0.3))),
                        child: const Icon(Icons.arrow_back_rounded,
                            color: Colors.white, size: 18)),
                    ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(color: Colors.white.withOpacity(0.2))),
                      child: Text('Final Step',
                          style: GoogleFonts.dmSans(
                              color: Colors.white, fontSize: 12,
                              fontWeight: FontWeight.w600)),
                    ),
                  ]),
                  const SizedBox(height: 16),
                  Row(children: List.generate(9, (i) => Expanded(
                    child: Container(
                      height: 4,
                      margin: EdgeInsets.only(right: i < 8 ? 4 : 0),
                      decoration: BoxDecoration(
                        color: AppTheme.tealMint,
                        borderRadius: BorderRadius.circular(2)),
                    ),
                  ))),
                  const SizedBox(height: 14),
                  Text('Review & Submit',
                      style: GoogleFonts.playfairDisplay(
                          fontSize: 30, fontWeight: FontWeight.w700,
                          color: Colors.white)),
                  const SizedBox(height: 4),
                  Text('Almost there! Final review',
                      style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.6), fontSize: 13)),
                ]),
              ),
              Expanded(
                child: AnimatedBuilder(
                  animation: _anim,
                  builder: (_, child) => Transform.translate(
                    offset: Offset(0, _slide.value),
                    child: Opacity(opacity: _fade.value, child: child),
                  ),
                  child: DraggableScrollableSheet(
                    initialChildSize: 0.85,
                    minChildSize: 0.45,
                    maxChildSize: 0.96,
                    expand: true,
                    builder: (ctx, scrollCtrl) => _buildSheet(scrollCtrl),
                  ),
                ),
              ),
            ]),
          ),
        ),
      ),
    );
  }

  Widget _buildSheet(ScrollController scrollCtrl) {
    return Container(
      width: double.infinity,
      decoration: const BoxDecoration(
        color: AppTheme.bg,
        borderRadius: BorderRadius.vertical(top: Radius.circular(32)),
      ),
      child: Column(children: [
        const SizedBox(height: 12),
        Center(child: Container(width: 40, height: 4,
          decoration: BoxDecoration(color: AppTheme.border,
              borderRadius: BorderRadius.circular(2)))),
        Expanded(
          child: SingleChildScrollView(
            controller: scrollCtrl,
            padding: const EdgeInsets.fromLTRB(20, 14, 20, 32),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start,
                children: [
              // Hostel summary card
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [AppTheme.teal.withOpacity(0.08),
                      AppTheme.teal.withOpacity(0.03)],
                    begin: Alignment.topLeft, end: Alignment.bottomRight),
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: AppTheme.teal.withOpacity(0.2), width: 0.5)),
                child: Row(children: [
                  Container(
                    width: 44, height: 44,
                    decoration: BoxDecoration(
                        color: AppTheme.teal.withOpacity(0.12),
                        borderRadius: BorderRadius.circular(12)),
                    child: const Icon(Icons.apartment_rounded, color: AppTheme.teal, size: 22)),
                  const SizedBox(width: 12),
                  Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(widget.hostelName, style: GoogleFonts.dmSans(
                        fontSize: 15, fontWeight: FontWeight.w700,
                        color: AppTheme.textPrimary)),
                    Text('${widget.hostelType}  ·  ${widget.city}',
                        style: GoogleFonts.dmSans(fontSize: 12, color: AppTheme.teal)),
                  ]),
                ]),
              ),
              const SizedBox(height: 14),

              // Completed steps
              Row(children: [
                const Icon(Icons.fact_check_rounded, size: 14, color: AppTheme.teal),
                const SizedBox(width: 6),
                Text('Completed Steps', style: GoogleFonts.dmSans(
                    fontSize: 13, fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary)),
              ]),
              const SizedBox(height: 8),
              ...['Basic Information', 'Map Location', 'Room & Bed Details',
                'Facilities', 'Mess & Food', 'Safety & Security',
                'Rules & Policies', 'Photos & Videos',
              ].map((s) => Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Row(children: [
                  Container(width: 20, height: 20,
                    decoration: const BoxDecoration(
                        color: AppTheme.green, shape: BoxShape.circle),
                    child: const Icon(Icons.check_rounded,
                        color: Colors.white, size: 12)),
                  const SizedBox(width: 10),
                  Text(s, style: GoogleFonts.dmSans(
                      fontSize: 13, color: AppTheme.textPrimary)),
                ]),
              )),
              const SizedBox(height: 16),
              Divider(height: 1, color: AppTheme.border, thickness: 0.5),
              const SizedBox(height: 14),

              // Summary fields
              _field(_summaryCtrl, 'Add a short description for students...',
                  Icons.notes_rounded, maxLines: 3),
              const SizedBox(height: 10),
              _field(_facilitiesCtrl, 'Key facilities (WiFi, Meals, AC...)',
                  Icons.checklist_rounded, maxLines: 2),
              const SizedBox(height: 10),
              _field(_rulesCtrl, 'Key rules (Curfew 10PM, No guests...)',
                  Icons.rule_rounded, maxLines: 2),
              const SizedBox(height: 16),

              // Terms checkbox
              GestureDetector(
                onTap: () => setState(() => _agreed = !_agreed),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 180),
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: _agreed ? AppTheme.teal.withOpacity(0.05) : AppTheme.card,
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(
                        color: _agreed ? AppTheme.teal : AppTheme.border,
                        width: 0.5)),
                  child: Row(children: [
                    AnimatedContainer(
                      duration: const Duration(milliseconds: 180),
                      width: 22, height: 22,
                      decoration: BoxDecoration(
                        color: _agreed ? AppTheme.teal : AppTheme.card,
                        borderRadius: BorderRadius.circular(6),
                        border: Border.all(
                            color: _agreed ? AppTheme.teal : AppTheme.border)),
                      child: _agreed ? const Icon(Icons.check_rounded,
                          color: Colors.white, size: 14) : null,
                    ),
                    const SizedBox(width: 12),
                    Expanded(child: Text(
                      'I confirm all information is accurate and agree to StayBuddy\'s Terms & Conditions',
                      style: GoogleFonts.dmSans(fontSize: 12,
                          color: AppTheme.textSecondary, height: 1.4))),
                  ]),
                ),
              ),
              const SizedBox(height: 18),

              // Submit button
              SizedBox(
                width: double.infinity, height: 52,
                child: ElevatedButton(
                  onPressed: _submitting ? null : _submit,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.teal, foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16))),
                  child: _submitting
                      ? const SizedBox(width: 20, height: 20,
                          child: CircularProgressIndicator(
                              strokeWidth: 2, color: Colors.white))
                      : Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                          const Icon(Icons.check_circle_rounded, size: 18),
                          const SizedBox(width: 10),
                          Text('Submit Registration',
                              style: GoogleFonts.dmSans(
                                  fontSize: 15, fontWeight: FontWeight.w600)),
                        ]),
                ),
              ),
            ]),
          ),
        ),
      ]),
    );
  }

  Widget _field(TextEditingController ctrl, String hint, IconData icon,
      {int maxLines = 1}) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.card, borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.border, width: 0.5)),
      child: TextField(
        controller: ctrl, maxLines: maxLines,
        style: GoogleFonts.dmSans(fontSize: 14, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13),
          prefixIcon: Icon(icon, size: 18, color: AppTheme.textMuted),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14)),
      ),
    );
  }
}