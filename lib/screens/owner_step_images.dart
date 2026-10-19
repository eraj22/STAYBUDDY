import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:image_picker/image_picker.dart';

import '../api.dart';
import '../models/owner_attendance_boundary_draft.dart';
import '../models/owner_room_setup_draft.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_hostel_success_screen.dart';

// =============================================================================
// COLORS
// =============================================================================

const Color _darkTeal = Color(0xFF04383B);
const Color _deepTeal = Color(0xFF02292C);
const Color _fieldTeal = Color(0xFF07585C);
const Color _danger = Color(0xFFE25A5A);
const Color _green = Color(0xFF54C49A);

// =============================================================================
// STEP 9 — PHOTOS & VIDEOS
// =============================================================================

class OwnerStepImages extends StatefulWidget {
  final String ownerEmail;
  final String ownerName;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;
  final double latitude;
  final double longitude;
  final int totalFloors;
  final int totalRooms;
  final int totalCapacity;
  final List<Map<String, dynamic>> roomTypes;
  final List<String> facilities;
  final bool messIncluded;
  final bool extraMessCharges;
  final String messCharges;
  final int mealsPerDay;
  final String mealType;
  final List<String> mealTimings;
  final List<String> menuItems;
  final bool attendanceSystem;
  final bool entryCardSystem;
  final bool cctvInstalled;
  final List<String> cctvAreas;
  final bool wardenOnsite;
  final List<String> guardShifts;
  final bool hasCurfew;
  final String curfewTime;
  final String attendanceType;
  final bool lateEntryAllowed;
  final bool parentAbsenceNotification;
  final String guestPolicy;
  final List<String> paymentMethods;
  final bool finePolicy;
  final String fineDetails;
  final List<String> additionalRules;

  const OwnerStepImages({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
    required this.latitude,
    required this.longitude,
    required this.totalFloors,
    required this.totalRooms,
    required this.totalCapacity,
    required this.roomTypes,
    required this.facilities,
    required this.messIncluded,
    required this.extraMessCharges,
    required this.messCharges,
    required this.mealsPerDay,
    required this.mealType,
    required this.mealTimings,
    required this.menuItems,
    required this.attendanceSystem,
    required this.entryCardSystem,
    required this.cctvInstalled,
    required this.cctvAreas,
    required this.wardenOnsite,
    required this.guardShifts,
    required this.hasCurfew,
    required this.curfewTime,
    required this.attendanceType,
    required this.lateEntryAllowed,
    required this.parentAbsenceNotification,
    required this.guestPolicy,
    required this.paymentMethods,
    required this.finePolicy,
    required this.fineDetails,
    required this.additionalRules,
  });

  @override
  State<OwnerStepImages> createState() => _OwnerStepImagesState();
}

class _OwnerStepImagesState extends State<OwnerStepImages>
    with SingleTickerProviderStateMixin {
  final ImagePicker _picker = ImagePicker();

  XFile? _exteriorImage;
  XFile? _roomImage;
  XFile? _commonImage;
  XFile? _walkthroughVideo;

  String? _error;

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  bool get _hasExterior => _exteriorImage != null;

  bool get _hasRoom => _roomImage != null;

  bool get _hasCommon => _commonImage != null;

  bool get _hasVideo => _walkthroughVideo != null;

  @override
  void initState() {
    super.initState();

    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 750),
    );

    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);

    _slide = Tween<double>(
      begin: 30,
      end: 0,
    ).animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));

    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    super.dispose();
  }

  // ===========================================================================
  // PICK FILES
  // ===========================================================================

  Future<void> _pickExterior() async {
    final file = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 80,
    );

    if (file == null || !mounted) {
      return;
    }

    setState(() {
      _exteriorImage = file;
      _error = null;
    });
  }

  Future<void> _pickRoom() async {
    final file = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 80,
    );

    if (file == null || !mounted) {
      return;
    }

    setState(() {
      _roomImage = file;
      _error = null;
    });
  }

  Future<void> _pickCommon() async {
    final file = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 80,
    );

    if (file == null || !mounted) {
      return;
    }

    setState(() {
      _commonImage = file;
      _error = null;
    });
  }

  Future<void> _pickVideo() async {
    final file = await _picker.pickVideo(
      source: ImageSource.gallery,
      maxDuration: const Duration(minutes: 2),
    );

    if (file == null || !mounted) {
      return;
    }

    setState(() {
      _walkthroughVideo = file;
      _error = null;
    });
  }

  // ===========================================================================
  // NEXT
  // ===========================================================================

  void _next() {
    if (!_hasExterior) {
      setState(() {
        _error = 'Please upload at least one hostel exterior photo.';
      });

      return;
    }

    setState(() {
      _error = null;
    });

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerStepReview(
          ownerEmail: widget.ownerEmail,
          ownerName: widget.ownerName,
          hostelName: widget.hostelName,
          hostelType: widget.hostelType,
          city: widget.city,
          address: widget.address,
          latitude: widget.latitude,
          longitude: widget.longitude,
          totalFloors: widget.totalFloors,
          totalRooms: widget.totalRooms,
          totalCapacity: widget.totalCapacity,
          roomTypes: widget.roomTypes,
          facilities: widget.facilities,
          messIncluded: widget.messIncluded,
          extraMessCharges: widget.extraMessCharges,
          messCharges: widget.messCharges,
          mealsPerDay: widget.mealsPerDay,
          mealType: widget.mealType,
          mealTimings: widget.mealTimings,
          menuItems: widget.menuItems,
          attendanceSystem: widget.attendanceSystem,
          entryCardSystem: widget.entryCardSystem,
          cctvInstalled: widget.cctvInstalled,
          cctvAreas: widget.cctvAreas,
          wardenOnsite: widget.wardenOnsite,
          guardShifts: widget.guardShifts,
          hasCurfew: widget.hasCurfew,
          curfewTime: widget.curfewTime,
          attendanceType: widget.attendanceType,
          lateEntryAllowed: widget.lateEntryAllowed,
          parentAbsenceNotification: widget.parentAbsenceNotification,
          guestPolicy: widget.guestPolicy,
          paymentMethods: widget.paymentMethods,
          finePolicy: widget.finePolicy,
          fineDetails: widget.fineDetails,
          additionalRules: widget.additionalRules,
        ),
      ),
    );
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.28,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildHeader(),

                const SizedBox(height: 14),

                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                    child: AnimatedBuilder(
                      animation: _anim,
                      builder: (context, child) {
                        return Transform.translate(
                          offset: Offset(0, _slide.value),
                          child: Opacity(opacity: _fade.value, child: child),
                        );
                      },
                      child: _buildGlassPanel(),
                    ),
                  ),
                ),
              ],
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
      padding: const EdgeInsets.fromLTRB(18, 12, 18, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _backButton(),

              const Spacer(),

              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 7,
                ),
                decoration: BoxDecoration(
                  color: _deepTeal.withOpacity(0.62),
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(
                    color: AppTheme.tealMint.withOpacity(0.28),
                  ),
                ),
                child: Text(
                  'Step 9 of 9',
                  style: GoogleFonts.dmSans(
                    color: AppTheme.tealMint,
                    fontSize: 10.5,
                    fontWeight: FontWeight.w700,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 16),

          _progressBar(),

          const SizedBox(height: 16),

          Text(
            'Hostel Photos\n& Videos',
            style: GoogleFonts.playfairDisplay(
              color: Colors.white,
              fontSize: 33,
              height: 1.02,
              fontWeight: FontWeight.w700,
            ),
          ),

          const SizedBox(height: 10),

          Text(
            'Show students what they can expect from your hostel.',
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.70),
              fontSize: 12.5,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _backButton() {
    return InkWell(
      onTap: () => Navigator.maybePop(context),
      borderRadius: BorderRadius.circular(50),
      child: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: _darkTeal.withOpacity(0.62),
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white.withOpacity(0.22)),
        ),
        child: const Icon(
          Icons.arrow_back_rounded,
          color: Colors.white,
          size: 20,
        ),
      ),
    );
  }

  Widget _progressBar() {
    return Row(
      children: List.generate(
        9,
        (index) => Expanded(
          child: Container(
            height: 4,
            margin: EdgeInsets.only(right: index < 8 ? 5 : 0),
            decoration: BoxDecoration(
              color: AppTheme.tealMint,
              borderRadius: BorderRadius.circular(3),
            ),
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // GLASS PANEL
  // ===========================================================================

  Widget _buildGlassPanel() {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF06484B).withOpacity(0.88),
            const Color(0xFF023336).withOpacity(0.88),
            const Color(0xFF021F22).withOpacity(0.92),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.24)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.30),
            blurRadius: 26,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
        padding: const EdgeInsets.fromLTRB(18, 18, 18, 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Container(
                  width: 46,
                  height: 46,
                  decoration: BoxDecoration(
                    color: AppTheme.teal.withOpacity(0.24),
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: const Icon(
                    Icons.photo_library_outlined,
                    color: AppTheme.tealMint,
                    size: 22,
                  ),
                ),

                const SizedBox(width: 12),

                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Add Hostel Media',
                        style: GoogleFonts.playfairDisplay(
                          color: Colors.white,
                          fontSize: 21,
                          fontWeight: FontWeight.w700,
                        ),
                      ),
                      const SizedBox(height: 3),
                      Text(
                        'Upload clear and recent photos',
                        style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.55),
                          fontSize: 10,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),

            const SizedBox(height: 18),

            _tipCard(),

            const SizedBox(height: 16),

            _uploadCard(
              icon: Icons.home_rounded,
              title: 'Hostel Exterior',
              subtitle: 'Main building front view',
              uploaded: _hasExterior,
              requiredFile: true,
              fileName: _exteriorImage?.name,
              onTap: _pickExterior,
              onRemove: () {
                setState(() {
                  _exteriorImage = null;
                });
              },
            ),

            const SizedBox(height: 10),

            _uploadCard(
              icon: Icons.bed_rounded,
              title: 'Room Photos',
              subtitle: 'Interior room view',
              uploaded: _hasRoom,
              fileName: _roomImage?.name,
              onTap: _pickRoom,
              onRemove: () {
                setState(() {
                  _roomImage = null;
                });
              },
            ),

            const SizedBox(height: 10),

            _uploadCard(
              icon: Icons.weekend_rounded,
              title: 'Common Areas',
              subtitle: 'Lounge, dining and corridors',
              uploaded: _hasCommon,
              fileName: _commonImage?.name,
              onTap: _pickCommon,
              onRemove: () {
                setState(() {
                  _commonImage = null;
                });
              },
            ),

            const SizedBox(height: 10),

            _uploadCard(
              icon: Icons.videocam_rounded,
              title: 'Walkthrough Video',
              subtitle: 'Maximum duration: 2 minutes',
              uploaded: _hasVideo,
              optional: true,
              fileName: _walkthroughVideo?.name,
              onTap: _pickVideo,
              onRemove: () {
                setState(() {
                  _walkthroughVideo = null;
                });
              },
            ),

            if (_error != null) ...[
              const SizedBox(height: 14),
              _errorBox(_error!),
            ],

            const SizedBox(height: 18),

            SizedBox(
              width: double.infinity,
              height: 50,
              child: ElevatedButton(
                onPressed: _next,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.teal,
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      'Review & Submit',
                      style: GoogleFonts.dmSans(
                        fontSize: 14.5,
                        fontWeight: FontWeight.w800,
                      ),
                    ),
                    const SizedBox(width: 9),
                    const Icon(Icons.arrow_forward_rounded, size: 18),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // TIP
  // ===========================================================================

  Widget _tipCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: AppTheme.teal.withOpacity(0.15),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.20)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(
            Icons.tips_and_updates_outlined,
            color: AppTheme.tealMint,
            size: 18,
          ),

          const SizedBox(width: 10),

          Expanded(
            child: Text(
              'Good photos make your hostel profile more trustworthy. An exterior photo is required before continuing.',
              style: GoogleFonts.dmSans(
                color: Colors.white.withOpacity(0.72),
                fontSize: 10.5,
                height: 1.45,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // UPLOAD CARD
  // ===========================================================================

  Widget _uploadCard({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool uploaded,
    required VoidCallback onTap,
    required VoidCallback onRemove,
    String? fileName,
    bool optional = false,
    bool requiredFile = false,
  }) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      width: double.infinity,
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: uploaded
            ? AppTheme.teal.withOpacity(0.20)
            : Colors.black.withOpacity(0.20),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: uploaded
              ? AppTheme.tealMint.withOpacity(0.55)
              : Colors.white.withOpacity(0.12),
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                width: 43,
                height: 43,
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(uploaded ? 0.35 : 0.18),
                  borderRadius: BorderRadius.circular(13),
                ),
                child: Icon(
                  uploaded ? Icons.check_circle_rounded : icon,
                  color: AppTheme.tealMint,
                  size: 21,
                ),
              ),

              const SizedBox(width: 11),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Flexible(
                          child: Text(
                            title,
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.dmSans(
                              color: Colors.white,
                              fontSize: 11.5,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ),

                        if (optional) ...[
                          const SizedBox(width: 6),
                          _smallTag('Optional'),
                        ],

                        if (requiredFile) ...[
                          const SizedBox(width: 6),
                          _smallTag('Required'),
                        ],
                      ],
                    ),

                    const SizedBox(height: 3),

                    Text(
                      uploaded && fileName != null ? fileName : subtitle,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.50),
                        fontSize: 8.8,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(width: 8),

              InkWell(
                onTap: uploaded ? onRemove : onTap,
                borderRadius: BorderRadius.circular(11),
                child: Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 7,
                  ),
                  decoration: BoxDecoration(
                    color: uploaded
                        ? _danger.withOpacity(0.15)
                        : AppTheme.teal.withOpacity(0.25),
                    borderRadius: BorderRadius.circular(11),
                  ),
                  child: Text(
                    uploaded ? 'Remove' : 'Upload',
                    style: GoogleFonts.dmSans(
                      color: uploaded
                          ? const Color(0xFFFF8B8B)
                          : AppTheme.tealMint,
                      fontSize: 9,
                      fontWeight: FontWeight.w800,
                    ),
                  ),
                ),
              ),
            ],
          ),

          if (uploaded) ...[
            const SizedBox(height: 10),

            SizedBox(
              width: double.infinity,
              height: 34,
              child: OutlinedButton.icon(
                onPressed: onTap,
                icon: const Icon(Icons.swap_horiz_rounded, size: 15),
                label: const Text('Choose Different File'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppTheme.tealMint,
                  side: BorderSide(color: AppTheme.tealMint.withOpacity(0.25)),
                  textStyle: GoogleFonts.dmSans(
                    fontSize: 9.5,
                    fontWeight: FontWeight.w700,
                  ),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(11),
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _smallTag(String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 3),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.08),
        borderRadius: BorderRadius.circular(7),
      ),
      child: Text(
        text,
        style: GoogleFonts.dmSans(
          color: Colors.white.withOpacity(0.60),
          fontSize: 6.8,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

// =============================================================================
// FINAL REVIEW
// =============================================================================

class OwnerStepReview extends StatefulWidget {
  final String ownerEmail;
  final String ownerName;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;
  final double latitude;
  final double longitude;
  final int totalFloors;
  final int totalRooms;
  final int totalCapacity;
  final List<Map<String, dynamic>> roomTypes;
  final List<String> facilities;
  final bool messIncluded;
  final bool extraMessCharges;
  final String messCharges;
  final int mealsPerDay;
  final String mealType;
  final List<String> mealTimings;
  final List<String> menuItems;
  final bool attendanceSystem;
  final bool entryCardSystem;
  final bool cctvInstalled;
  final List<String> cctvAreas;
  final bool wardenOnsite;
  final List<String> guardShifts;
  final bool hasCurfew;
  final String curfewTime;
  final String attendanceType;
  final bool lateEntryAllowed;
  final bool parentAbsenceNotification;
  final String guestPolicy;
  final List<String> paymentMethods;
  final bool finePolicy;
  final String fineDetails;
  final List<String> additionalRules;

  const OwnerStepReview({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
    required this.latitude,
    required this.longitude,
    required this.totalFloors,
    required this.totalRooms,
    required this.totalCapacity,
    required this.roomTypes,
    required this.facilities,
    required this.messIncluded,
    required this.extraMessCharges,
    required this.messCharges,
    required this.mealsPerDay,
    required this.mealType,
    required this.mealTimings,
    required this.menuItems,
    required this.attendanceSystem,
    required this.entryCardSystem,
    required this.cctvInstalled,
    required this.cctvAreas,
    required this.wardenOnsite,
    required this.guardShifts,
    required this.hasCurfew,
    required this.curfewTime,
    required this.attendanceType,
    required this.lateEntryAllowed,
    required this.parentAbsenceNotification,
    required this.guestPolicy,
    required this.paymentMethods,
    required this.finePolicy,
    required this.fineDetails,
    required this.additionalRules,
  });

  @override
  State<OwnerStepReview> createState() => _OwnerStepReviewState();
}

class _OwnerStepReviewState extends State<OwnerStepReview>
    with SingleTickerProviderStateMixin {
  final _summaryCtrl = TextEditingController();

  final _facilitiesCtrl = TextEditingController();

  final _rulesCtrl = TextEditingController();

  bool _agreed = false;
  bool _submitting = false;

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
      begin: 30,
      end: 0,
    ).animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));

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

  // ===========================================================================
  // SUBMIT
  // ===========================================================================

  Future<void> _submit() async {
    FocusScope.of(context).unfocus();

    if (!_agreed) {
      _showToast('Please agree to the terms first.', error: true);
      return;
    }

    setState(() {
      _submitting = true;
    });

    try {
      final descriptionParts = <String>[
        if (_summaryCtrl.text.trim().isNotEmpty)
          'Profile Summary: ${_summaryCtrl.text.trim()}',
        'Mess Included: ${widget.messIncluded ? "Yes" : "No"}',
        'Extra Mess Charges: ${widget.extraMessCharges ? "Yes" : "No"}',
        if (widget.messCharges.trim().isNotEmpty)
          'Mess Charges: ${widget.messCharges.trim()}',
        'Meals Per Day: ${widget.mealsPerDay}',
        if (widget.mealType.trim().isNotEmpty) 'Meal Type: ${widget.mealType}',
        if (widget.mealTimings.isNotEmpty)
          'Meal Timings: ${widget.mealTimings.join(", ")}',
        if (widget.menuItems.isNotEmpty)
          'Menu Items: ${widget.menuItems.join(", ")}',
        'Attendance System: ${widget.attendanceSystem ? "Yes" : "No"}',
        'Entry Card System: ${widget.entryCardSystem ? "Yes" : "No"}',
        'CCTV Installed: ${widget.cctvInstalled ? "Yes" : "No"}',
        if (widget.cctvAreas.isNotEmpty)
          'CCTV Areas: ${widget.cctvAreas.join(", ")}',
        'Warden Onsite: ${widget.wardenOnsite ? "Yes" : "No"}',
        if (widget.guardShifts.isNotEmpty)
          'Guard Shifts: ${widget.guardShifts.join(", ")}',
        'Curfew: ${widget.hasCurfew ? "Yes" : "No"}',
        if (widget.hasCurfew && widget.curfewTime.trim().isNotEmpty)
          'Curfew Time: ${widget.curfewTime.trim()}',
        if (widget.attendanceType.trim().isNotEmpty)
          'Attendance Type: ${widget.attendanceType.trim()}',
        'Late Entry Allowed: ${widget.lateEntryAllowed ? "Yes" : "No"}',
        'Parent Absence Notification: ${widget.parentAbsenceNotification ? "Yes" : "No"}',
        if (widget.guestPolicy.trim().isNotEmpty)
          'Guest Policy: ${widget.guestPolicy.trim()}',
        if (widget.paymentMethods.isNotEmpty)
          'Payment Methods: ${widget.paymentMethods.join(", ")}',
        'Fine Policy: ${widget.finePolicy ? "Yes" : "No"}',
        if (widget.fineDetails.trim().isNotEmpty)
          'Fine Details: ${widget.fineDetails.trim()}',
        if (widget.additionalRules.isNotEmpty)
          'Additional Rules: ${widget.additionalRules.join(", ")}',
        if (_facilitiesCtrl.text.trim().isNotEmpty)
          'Additional Facilities: ${_facilitiesCtrl.text.trim()}',
        if (_rulesCtrl.text.trim().isNotEmpty)
          'Additional Rules Notes: ${_rulesCtrl.text.trim()}',
      ];

      final description = descriptionParts.join('\n');

      await Api().registerHostel(
        name: widget.hostelName,
        hostelType: widget.hostelType,
        city: widget.city,
        address: widget.address,
        latitude: widget.latitude,
        longitude: widget.longitude,
        totalCapacity: widget.totalCapacity,
        floors: widget.totalFloors,
        totalRooms: widget.totalRooms,
        amenities: widget.facilities,
        facilities: widget.facilities,
        rules: [
          ...widget.additionalRules,
          if (_rulesCtrl.text.trim().isNotEmpty) _rulesCtrl.text.trim(),
        ],
        messInfo: {
          'mess_included': widget.messIncluded,
          'extra_mess_charges': widget.extraMessCharges,
          'mess_charges': widget.messCharges,
          'meals_per_day': widget.mealsPerDay,
          'meal_type': widget.mealType,
          'meal_timings': widget.mealTimings,
          'menu_items': widget.menuItems,
          'attendance_system': widget.attendanceSystem,
          'entry_card_system': widget.entryCardSystem,
          'cctv_installed': widget.cctvInstalled,
          'cctv_areas': widget.cctvAreas,
          'warden_onsite': widget.wardenOnsite,
          'guard_shifts': widget.guardShifts,
          'has_curfew': widget.hasCurfew,
          'curfew_time': widget.curfewTime,
          'attendance_type': widget.attendanceType,
          'late_entry_allowed': widget.lateEntryAllowed,
          'parent_absence_notification': widget.parentAbsenceNotification,
          'guest_policy': widget.guestPolicy,
          'payment_methods': widget.paymentMethods,
          'fine_policy': widget.finePolicy,
          'fine_details': widget.fineDetails,
        },
        description: description,
        roomTypes: widget.roomTypes,
        attendanceBoundary: OwnerAttendanceBoundaryDraft.points,
        rooms: OwnerRoomSetupDraft.rooms,
      );

      OwnerAttendanceBoundaryDraft.clear();
      OwnerRoomSetupDraft.clear();

      if (!mounted) return;

      setState(() {
        _submitting = false;
      });

      Navigator.pushReplacement(
        context,
        MaterialPageRoute(
          builder: (_) => OwnerHostelSuccessScreen(
            ownerName: widget.ownerName,
            ownerEmail: widget.ownerEmail,
            hostelName: widget.hostelName,
            hostelType: widget.hostelType,
            city: widget.city,
            address: widget.address,
          ),
        ),
      );
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _submitting = false;
      });

      _showToast('Hostel registration failed: $e', error: true);
    }
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    final keyboardHeight = MediaQuery.viewInsetsOf(context).bottom;

    return ResponsiveShell(
      child: Scaffold(
        resizeToAvoidBottomInset: true,
        backgroundColor: _deepTeal,
        body: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [_darkTeal, _deepTeal, Color(0xFF052C2F)],
            ),
          ),
          child: SafeArea(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _buildHeader(),
                const SizedBox(height: 14),
                Expanded(
                  child: Padding(
                    padding: EdgeInsets.fromLTRB(
                      16,
                      0,
                      16,
                      16 + keyboardHeight,
                    ),
                    child: AnimatedBuilder(
                      animation: _anim,
                      builder: (context, child) => Transform.translate(
                        offset: Offset(0, _slide.value),
                        child: Opacity(opacity: _fade.value, child: child),
                      ),
                      child: _buildReviewPanel(),
                    ),
                  ),
                ),
              ],
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
      padding: const EdgeInsets.fromLTRB(18, 12, 18, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              InkWell(
                onTap: () => Navigator.maybePop(context),
                borderRadius: BorderRadius.circular(50),
                child: Container(
                  width: 42,
                  height: 42,
                  decoration: BoxDecoration(
                    color: _darkTeal.withOpacity(0.62),
                    shape: BoxShape.circle,
                    border: Border.all(color: Colors.white.withOpacity(0.22)),
                  ),
                  child: const Icon(
                    Icons.arrow_back_rounded,
                    color: Colors.white,
                    size: 20,
                  ),
                ),
              ),

              const Spacer(),

              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 7,
                ),
                decoration: BoxDecoration(
                  color: _deepTeal.withOpacity(0.65),
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(
                    color: AppTheme.tealMint.withOpacity(0.30),
                  ),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.verified_rounded,
                      color: AppTheme.tealMint,
                      size: 13,
                    ),
                    const SizedBox(width: 5),
                    Text(
                      'Final Review',
                      style: GoogleFonts.dmSans(
                        color: AppTheme.tealMint,
                        fontSize: 10.5,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 16),

          Row(
            children: List.generate(
              9,
              (index) => Expanded(
                child: Container(
                  height: 4,
                  margin: EdgeInsets.only(right: index < 8 ? 5 : 0),
                  decoration: BoxDecoration(
                    color: AppTheme.tealMint,
                    borderRadius: BorderRadius.circular(3),
                  ),
                ),
              ),
            ),
          ),

          const SizedBox(height: 16),

          Text(
            'Review &\nSubmit',
            style: GoogleFonts.playfairDisplay(
              fontSize: 33,
              height: 1.02,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),

          const SizedBox(height: 10),

          Text(
            'Check everything before publishing your hostel.',
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.70),
              fontSize: 12.5,
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // REVIEW PANEL
  // ===========================================================================

  Widget _buildReviewPanel() {
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF06484B).withOpacity(0.90),
            const Color(0xFF023336).withOpacity(0.90),
            const Color(0xFF021F22).withOpacity(0.94),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.24)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.30),
            blurRadius: 26,
            offset: const Offset(0, 12),
          ),
        ],
      ),
      child: SingleChildScrollView(
        physics: const BouncingScrollPhysics(),
        keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
        padding: const EdgeInsets.fromLTRB(18, 18, 18, 24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _hostelSummary(),

            const SizedBox(height: 18),

            _sectionTitle(Icons.fact_check_outlined, 'Completed Steps'),

            const SizedBox(height: 10),

            _completedSteps(),

            const SizedBox(height: 18),

            _sectionTitle(Icons.edit_note_rounded, 'Final Profile Details'),

            const SizedBox(height: 10),

            _field(
              controller: _summaryCtrl,
              hint: 'Short description for students...',
              icon: Icons.notes_rounded,
              maxLines: 3,
            ),

            const SizedBox(height: 10),

            _field(
              controller: _facilitiesCtrl,
              hint: 'Key facilities e.g. WiFi, meals, AC...',
              icon: Icons.checklist_rounded,
              maxLines: 2,
            ),

            const SizedBox(height: 10),

            _field(
              controller: _rulesCtrl,
              hint: 'Key rules e.g. curfew 10 PM...',
              icon: Icons.rule_rounded,
              maxLines: 2,
            ),

            const SizedBox(height: 16),

            _termsBox(),

            const SizedBox(height: 18),

            SizedBox(
              width: double.infinity,
              height: 51,
              child: ElevatedButton(
                onPressed: _submitting ? null : _submit,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.teal,
                  disabledBackgroundColor: AppTheme.teal.withOpacity(0.50),
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                ),
                child: _submitting
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
                          const Icon(
                            Icons.check_circle_outline_rounded,
                            size: 18,
                          ),
                          const SizedBox(width: 9),
                          Text(
                            'Submit Registration',
                            style: GoogleFonts.dmSans(
                              fontSize: 14.5,
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ],
                      ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // HOSTEL SUMMARY
  // ===========================================================================

  Widget _hostelSummary() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.teal.withOpacity(0.17),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.20)),
      ),
      child: Row(
        children: [
          Container(
            width: 46,
            height: 46,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.30),
              borderRadius: BorderRadius.circular(14),
            ),
            child: const Icon(
              Icons.apartment_rounded,
              color: AppTheme.tealMint,
              size: 22,
            ),
          ),

          const SizedBox(width: 12),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.hostelName,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 13,
                    fontWeight: FontWeight.w800,
                  ),
                ),

                const SizedBox(height: 3),

                Text(
                  '${_formatType(widget.hostelType)} · ${widget.city}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: AppTheme.tealMint,
                    fontSize: 9.5,
                    fontWeight: FontWeight.w600,
                  ),
                ),

                const SizedBox(height: 3),

                Text(
                  widget.address,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white.withOpacity(0.48),
                    fontSize: 8.5,
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(width: 6),

          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 5),
            decoration: BoxDecoration(
              color: _green.withOpacity(0.15),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Text(
              'Ready',
              style: GoogleFonts.dmSans(
                color: _green,
                fontSize: 8,
                fontWeight: FontWeight.w800,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // COMPLETED STEPS
  // ===========================================================================

  Widget _completedSteps() {
    const steps = [
      'Owner Profile',
      'Email Verification',
      'Hostel Information',
      'Map Location',
      'Room & Bed Details',
      'Facilities & Amenities',
      'Mess & Food',
      'Safety & Rules',
      'Photos & Videos',
    ];

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.18),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.10)),
      ),
      child: Column(
        children: steps.asMap().entries.map((entry) {
          final last = entry.key == steps.length - 1;

          return Container(
            padding: const EdgeInsets.symmetric(vertical: 7),
            decoration: BoxDecoration(
              border: last
                  ? null
                  : Border(
                      bottom: BorderSide(color: Colors.white.withOpacity(0.06)),
                    ),
            ),
            child: Row(
              children: [
                Container(
                  width: 22,
                  height: 22,
                  decoration: const BoxDecoration(
                    color: _green,
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.check_rounded,
                    size: 13,
                    color: Colors.white,
                  ),
                ),

                const SizedBox(width: 10),

                Expanded(
                  child: Text(
                    entry.value,
                    style: GoogleFonts.dmSans(
                      color: Colors.white.withOpacity(0.76),
                      fontSize: 10,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ),
    );
  }

  // ===========================================================================
  // SECTION TITLE
  // ===========================================================================

  Widget _sectionTitle(IconData icon, String text) {
    return Row(
      children: [
        Icon(icon, color: AppTheme.tealMint, size: 16),
        const SizedBox(width: 7),
        Text(
          text,
          style: GoogleFonts.dmSans(
            color: Colors.white,
            fontSize: 11.5,
            fontWeight: FontWeight.w800,
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // WHITE INPUT FIELD
  // ===========================================================================

  Widget _field({
    required TextEditingController controller,
    required String hint,
    required IconData icon,
    int maxLines = 1,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.95),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white, width: 0.7),
      ),
      child: TextField(
        controller: controller,
        maxLines: maxLines,
        style: GoogleFonts.dmSans(
          color: const Color(0xFF183636),
          fontSize: 12.5,
          fontWeight: FontWeight.w500,
        ),
        cursorColor: AppTheme.teal,
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(
            color: const Color(0xFF82908F),
            fontSize: 11.5,
          ),
          prefixIcon: Icon(icon, size: 18, color: _fieldTeal),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 14,
            vertical: 14,
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // TERMS
  // ===========================================================================

  Widget _termsBox() {
    return InkWell(
      onTap: () {
        setState(() {
          _agreed = !_agreed;
        });
      },
      borderRadius: BorderRadius.circular(15),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 180),
        width: double.infinity,
        padding: const EdgeInsets.all(13),
        decoration: BoxDecoration(
          color: _agreed
              ? AppTheme.teal.withOpacity(0.18)
              : Colors.black.withOpacity(0.18),
          borderRadius: BorderRadius.circular(15),
          border: Border.all(
            color: _agreed
                ? AppTheme.tealMint.withOpacity(0.48)
                : Colors.white.withOpacity(0.10),
          ),
        ),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              width: 23,
              height: 23,
              decoration: BoxDecoration(
                color: _agreed ? AppTheme.teal : Colors.white.withOpacity(0.08),
                borderRadius: BorderRadius.circular(6),
                border: Border.all(
                  color: _agreed
                      ? AppTheme.tealMint
                      : Colors.white.withOpacity(0.24),
                ),
              ),
              child: _agreed
                  ? const Icon(
                      Icons.check_rounded,
                      color: Colors.white,
                      size: 15,
                    )
                  : null,
            ),

            const SizedBox(width: 11),

            Expanded(
              child: Text(
                'I confirm that all information provided is accurate and I agree to StayBuddy\'s Terms & Conditions.',
                style: GoogleFonts.dmSans(
                  color: Colors.white.withOpacity(0.68),
                  fontSize: 9.8,
                  height: 1.45,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // TOAST
  // ===========================================================================

  void _showToast(String message, {bool error = false}) {
    final messenger = ScaffoldMessenger.of(context);

    messenger
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: Colors.transparent,
          elevation: 0,
          margin: const EdgeInsets.fromLTRB(16, 0, 16, 18),
          padding: EdgeInsets.zero,
          content: Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            decoration: BoxDecoration(
              color: error ? const Color(0xFF8E3030) : AppTheme.teal,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Row(
              children: [
                Icon(
                  error
                      ? Icons.error_outline_rounded
                      : Icons.check_circle_outline_rounded,
                  color: Colors.white,
                  size: 18,
                ),
                const SizedBox(width: 9),
                Expanded(
                  child: Text(
                    message,
                    style: GoogleFonts.dmSans(
                      color: Colors.white,
                      fontSize: 10.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      );
  }

  String _formatType(String type) {
    final value = type.trim().toLowerCase();

    if (value == 'girls') {
      return 'Girls Hostel';
    }

    if (value == 'boys') {
      return 'Boys Hostel';
    }

    return type;
  }
}

// =============================================================================
// SHARED ERROR BOX
// =============================================================================

Widget _errorBox(String message) {
  return Container(
    width: double.infinity,
    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
    decoration: BoxDecoration(
      color: const Color(0xFF7F2525).withOpacity(0.45),
      borderRadius: BorderRadius.circular(13),
      border: Border.all(color: const Color(0xFFFF7777).withOpacity(0.38)),
    ),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Icon(
          Icons.error_outline_rounded,
          color: Color(0xFFFF9A9A),
          size: 17,
        ),

        const SizedBox(width: 8),

        Expanded(
          child: Text(
            message,
            style: GoogleFonts.dmSans(
              color: const Color(0xFFFFC0C0),
              fontSize: 10,
              height: 1.4,
            ),
          ),
        ),
      ],
    ),
  );
}
