import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'booking_screen.dart';

class StudentBookingInfoScreen extends StatefulWidget {
  final int hostelId;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;
  final int pricePerMonth;
  final double distanceKm;
  final double? rating;

  const StudentBookingInfoScreen({
    super.key,
    required this.hostelId,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
    required this.pricePerMonth,
    required this.distanceKm,
    this.rating,
  });

  @override
  State<StudentBookingInfoScreen> createState() =>
      _StudentBookingInfoScreenState();
}

class _StudentBookingInfoScreenState extends State<StudentBookingInfoScreen>
    with SingleTickerProviderStateMixin {
  int _tab = 0; // 0 = student info, 1 = parent info
  final _pageCtrl = PageController();

  // Student info
  final _sNameCtrl = TextEditingController();
  final _sEmailCtrl = TextEditingController();
  final _sPhoneCtrl = TextEditingController();
  final _sCnicCtrl = TextEditingController();
  final _sUniCtrl = TextEditingController();
  final _sDeptCtrl = TextEditingController();
  final _sSemCtrl = TextEditingController();
  String _sGender = 'Female';
  String _sOccupation = 'Student'; // Student or Working Professional
  String _sDegree = 'Bachelor'; // Bachelor or Master

  // Father / Guardian info
  final _pNameCtrl = TextEditingController();
  final _pPhoneCtrl = TextEditingController();
  final _pCnicCtrl = TextEditingController();
  String _pRelation = 'Father';
  final _pCityCtrl = TextEditingController();
  final _pEmailCtrl = TextEditingController();

  // Mother info (optional)
  bool _addMother = false;
  String _mRelation = 'Mother';
  final _mNameCtrl = TextEditingController();
  final _mPhoneCtrl = TextEditingController();
  final _mCnicCtrl = TextEditingController();
  final _mEmailCtrl = TextEditingController();

  late final AnimationController _anim;
  late final Animation<double> _fade;
  final Map<TextEditingController, String> _fieldErrors = {};

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 400),
    );
    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);
    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    _pageCtrl.dispose();
    for (final c in [
      _sNameCtrl,
      _sEmailCtrl,
      _sPhoneCtrl,
      _sCnicCtrl,
      _sUniCtrl,
      _sDeptCtrl,
      _sSemCtrl,
      _pNameCtrl,
      _pPhoneCtrl,
      _pCnicCtrl,
      _pCityCtrl,
      _pEmailCtrl,
      _mNameCtrl,
      _mPhoneCtrl,
      _mCnicCtrl,
      _mEmailCtrl,
    ]) {
      c.dispose();
    }
    super.dispose();
  }

  void _goToParentTab() {
    if (!_validateStudent()) {
      return;
    }
    setState(() => _tab = 1);
    _pageCtrl.animateToPage(
      1,
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeInOut,
    );
  }

  void _proceedToBooking() {
    if (!_validateParent()) {
      return;
    }
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => BookingScreen(
          hostelId: widget.hostelId,
          hostelName: widget.hostelName,
          city: widget.city,
          address: widget.address,
          studentName: _sNameCtrl.text.trim(),
          studentPhone: _sPhoneCtrl.text.trim(),
          studentEmail: _sEmailCtrl.text.trim(),
          studentCnic: _sCnicCtrl.text.trim(),
          studentUniversity: _sUniCtrl.text.trim(),
          studentDepartment: _sDeptCtrl.text.trim(),
          studentSemester: _sSemCtrl.text.trim(),
          studentGender: _sGender,
          studentOccupation: _sOccupation,
          studentDegree: _sDegree,
          parentName: _pNameCtrl.text.trim(),
          parentPhone: _pPhoneCtrl.text.trim(),
          parentRelation: _pRelation,
          parentCnic: _pCnicCtrl.text.trim(),
          parentCity: _pCityCtrl.text.trim(),
          parentEmail: _pEmailCtrl.text.trim(),
          secondaryGuardianName: _addMother ? _mNameCtrl.text.trim() : '',
          secondaryGuardianPhone: _addMother ? _mPhoneCtrl.text.trim() : '',
          secondaryGuardianCnic: _addMother ? _mCnicCtrl.text.trim() : '',
          secondaryGuardianRelation: _addMother ? _mRelation : '',
          secondaryGuardianEmail: _addMother ? _mEmailCtrl.text.trim() : '',
        ),
      ),
    );
  }

  bool _validEmail(String value) =>
      RegExp(r'^[^\s@]+@[^\s@]+\.[^\s@]+$').hasMatch(value.trim());

  bool _validPhone(String value) {
    final digits = value.replaceAll(RegExp(r'[^0-9]'), '');
    return digits.length >= 10 && digits.length <= 15;
  }

  bool _validCnic(String value) {
    final digits = value.replaceAll(RegExp(r'[^0-9]'), '');
    return digits.length == 13;
  }

  String? _inputError(String hint, String value) {
    final text = value.trim();
    final label = hint.replaceAll('*', '').trim();
    final lower = hint.toLowerCase();
    final required = hint.contains('*');

    if (required && text.isEmpty) return '$label is required';
    if (text.isEmpty) return null;

    if (lower.contains('full name')) {
      if (text.length < 2 ||
          !RegExp(r"^[^0-9]{2,80}$").hasMatch(text)) {
        return 'Use 2+ letters only';
      }
    }
    if (lower.contains('email') && !_validEmail(text)) {
      return 'Invalid email';
    }
    if (lower.contains('phone') && !_validPhone(text)) {
      return 'Use 10–15 digits';
    }
    if (lower.contains('cnic') && !_validCnic(text)) {
      return '13 digits required';
    }
    if (lower.contains('semester') || lower.contains('experience')) {
      final number = int.tryParse(text);
      if (number == null || number < 1 || number > 60) {
        return lower.contains('semester')
            ? 'Enter a valid semester or year'
            : 'Enter years from 1 to 60';
      }
    }
    if ((lower.contains('university') ||
            lower.contains('institution') ||
            lower.contains('department') ||
            lower.contains('organization') ||
            lower.contains('job title') ||
            lower.contains('city / address')) &&
        text.length < 2) {
      return 'Enter a valid $label';
    }
    return null;
  }

  void _validateLive(TextEditingController controller, String hint) {
    final error = _inputError(hint, controller.text);
    setState(() {
      if (error == null) {
        _fieldErrors.remove(controller);
      } else {
        _fieldErrors[controller] = error;
      }
    });
  }

  TextEditingValue _limitCnicToThirteenDigits(
    TextEditingValue oldValue,
    TextEditingValue newValue,
  ) {
    final digits = newValue.text.replaceAll(RegExp(r'[^0-9]'), '');
    final capped = digits.length > 13 ? digits.substring(0, 13) : digits;
    return TextEditingValue(
      text: capped,
      selection: TextSelection.collapsed(offset: capped.length),
    );
  }

  bool _validateStudent() {
    final errors = <TextEditingController, String>{};
    void validate(TextEditingController controller, String hint) {
      final error = _inputError(hint, controller.text);
      if (error != null) errors[controller] = error;
    }

    validate(_sNameCtrl, 'Full Name *');
    validate(_sEmailCtrl, 'Email Address');
    validate(_sPhoneCtrl, 'Phone Number *');
    validate(_sCnicCtrl, 'CNIC / B-Form Number');
    validate(
      _sUniCtrl,
      _sOccupation == 'Student'
          ? 'University / Institution'
          : 'Company / Organization',
    );
    validate(
      _sDeptCtrl,
      _sOccupation == 'Student' ? 'Department' : 'Job Title / Role',
    );
    validate(
      _sSemCtrl,
      _sOccupation == 'Student' ? 'Semester / Year' : 'Years of Experience',
    );
    setState(() {
      _fieldErrors
        ..clear()
        ..addAll(errors);
    });
    return errors.isEmpty;
  }

  bool _validateParent() {
    if (!_validateStudent()) return false;
    final errors = <TextEditingController, String>{};
    void validate(TextEditingController controller, String hint) {
      final error = _inputError(hint, controller.text);
      if (error != null) errors[controller] = error;
    }

    validate(_pNameCtrl, 'Full Name *');
    validate(_pPhoneCtrl, 'Phone Number *');
    validate(_pCnicCtrl, 'CNIC Number');
    validate(_pEmailCtrl, 'Email (optional)');
    validate(_pCityCtrl, 'City / Address');
    if (_addMother) {
      validate(_mNameCtrl, 'Full Name *');
      validate(_mPhoneCtrl, 'Phone Number *');
      validate(_mCnicCtrl, 'CNIC Number');
      validate(_mEmailCtrl, 'Email (optional)');
    }
    setState(() {
      _fieldErrors.removeWhere(
        (controller, _) => [
          _pNameCtrl,
          _pPhoneCtrl,
          _pCnicCtrl,
          _pCityCtrl,
          _pEmailCtrl,
          _mNameCtrl,
          _mPhoneCtrl,
          _mCnicCtrl,
          _mEmailCtrl,
        ].contains(controller),
      );
      _fieldErrors.addAll(errors);
    });
    return errors.isEmpty;
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.46,
          tintColor: AppTheme.tealDeep,
          tintOpacity: 0.20,
          cardMargin: const EdgeInsets.all(12),
          cardRadius: 30,
          cardColor: AppTheme.bg.withOpacity(0.97),
          child: Column(
            children: [
              _buildHeader(),
              _buildFlowIndicator(),
              // Tab bar
              _buildTabBar(),
              Expanded(
                child: PageView(
                  controller: _pageCtrl,
                  physics: const NeverScrollableScrollPhysics(),
                  children: [_buildStudentTab(), _buildParentTab()],
                ),
              ),
              _buildBottomBar(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFlowIndicator() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(18, 0, 18, 2),
      child: Row(
        children: [
          Expanded(child: _flowSegment(true)),
          const SizedBox(width: 6),
          Expanded(child: _flowSegment(_tab == 1)),
          const SizedBox(width: 6),
          Expanded(child: _flowSegment(false)),
        ],
      ),
    );
  }

  Widget _flowSegment(bool active) {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      height: 4,
      decoration: BoxDecoration(
        color: active ? AppTheme.teal : AppTheme.border,
        borderRadius: BorderRadius.circular(4),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────
  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [AppTheme.tealDeep, AppTheme.teal],
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          child: Column(
            children: [
              Row(
                children: [
                  GestureDetector(
                    onTap: () => Navigator.pop(context),
                    child: Container(
                      width: 38,
                      height: 38,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.15),
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: Colors.white.withOpacity(0.3),
                        ),
                      ),
                      child: const Icon(
                        Icons.arrow_back_rounded,
                        color: Colors.white,
                        size: 18,
                      ),
                    ),
                  ),
                  const SizedBox(width: 14),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Booking Information',
                          style: GoogleFonts.playfairDisplay(
                            fontSize: 20,
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ),
                        ),
                        Text(
                          'Fill your details to proceed',
                          style: GoogleFonts.dmSans(
                            fontSize: 11,
                            color: Colors.white60,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),

              // Hostel info bar
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.white.withOpacity(0.2)),
                ),
                child: Row(
                  children: [
                    Container(
                      width: 40,
                      height: 40,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: const Icon(
                        Icons.apartment_rounded,
                        color: Colors.white,
                        size: 20,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            widget.hostelName,
                            style: GoogleFonts.dmSans(
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                              color: Colors.white,
                            ),
                            maxLines: 1,
                            overflow: TextOverflow.ellipsis,
                          ),
                          Text(
                            '${widget.city}  ·  ${widget.distanceKm.toStringAsFixed(1)} km',
                            style: GoogleFonts.dmSans(
                              fontSize: 11,
                              color: Colors.white60,
                            ),
                          ),
                        ],
                      ),
                    ),
                    if (widget.pricePerMonth > 0)
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 5,
                        ),
                        decoration: BoxDecoration(
                          color: Colors.white.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          'PKR ${widget.pricePerMonth}/mo',
                          style: GoogleFonts.dmSans(
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ),
                        ),
                      ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ── Tab bar ───────────────────────────────────────────────────────────────
  Widget _buildTabBar() {
    return Container(
      color: AppTheme.bg,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      child: Row(
        children: [
          Expanded(
            child: _tabBtn(
              0,
              'Student Info',
              Icons.school_rounded,
              AppTheme.teal,
            ),
          ),
          const SizedBox(width: 10),
          Expanded(
            child: _tabBtn(
              1,
              'Parent Info',
              Icons.family_restroom_rounded,
              AppTheme.teal,
            ),
          ),
        ],
      ),
    );
  }

  Widget _tabBtn(int idx, String label, IconData icon, Color color) {
    final sel = _tab == idx;
    return GestureDetector(
      onTap: idx == 1
          ? _goToParentTab
          : () {
              setState(() => _tab = 0);
              _pageCtrl.animateToPage(
                0,
                duration: const Duration(milliseconds: 300),
                curve: Curves.easeInOut,
              );
            },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(vertical: 10),
        decoration: BoxDecoration(
          color: sel ? color : AppTheme.card,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: sel ? color : AppTheme.border, width: 0.5),
        ),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              icon,
              size: 16,
              color: sel ? Colors.white : AppTheme.textMuted,
            ),
            const SizedBox(width: 7),
            Text(
              label,
              style: GoogleFonts.dmSans(
                fontSize: 13,
                fontWeight: FontWeight.w600,
                color: sel ? Colors.white : AppTheme.textMuted,
              ),
            ),
            if (sel) ...[
              const SizedBox(width: 6),
              Container(
                width: 6,
                height: 6,
                decoration: const BoxDecoration(
                  color: Colors.white,
                  shape: BoxShape.circle,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  // ── Student Tab ───────────────────────────────────────────────────────────
  Widget _buildStudentTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      child: _formSurface(
        child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (_fieldErrors.isNotEmpty) ...[
            _validationBanner('Please correct the highlighted fields before continuing.'),
            const SizedBox(height: 12),
          ],
          _sectionHeader(
            'Personal Information',
            Icons.person_rounded,
            AppTheme.teal,
          ),
          const SizedBox(height: 10),
          _field(_sNameCtrl, 'Full Name *', Icons.person_outline_rounded),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: _field(
                  _sEmailCtrl,
                  'Email Address',
                  Icons.email_outlined,
                  type: TextInputType.emailAddress,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _field(
                  _sPhoneCtrl,
                  'Phone Number *',
                  Icons.phone_rounded,
                  type: TextInputType.phone,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          _field(
            _sCnicCtrl,
            'CNIC / B-Form Number',
            Icons.credit_card_rounded,
            type: TextInputType.number,
          ),
          const SizedBox(height: 10),

          // Gender selector
          _label('Gender', Icons.wc_rounded),
          const SizedBox(height: 6),
          Row(
            children: ['Female', 'Male'].map((g) {
              final sel = _sGender == g;
              const color = AppTheme.teal;
              return Expanded(
                child: Padding(
                  padding: EdgeInsets.only(right: g == 'Female' ? 8 : 0),
                  child: GestureDetector(
                    onTap: () => setState(() => _sGender = g),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 160),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      decoration: BoxDecoration(
                        color: sel ? color : AppTheme.card,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: sel ? color : AppTheme.border,
                          width: 0.5,
                        ),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            g == 'Female'
                                ? Icons.female_rounded
                                : Icons.male_rounded,
                            size: 18,
                            color: sel ? Colors.white : AppTheme.textMuted,
                          ),
                          const SizedBox(width: 6),
                          Text(
                            g,
                            style: GoogleFonts.dmSans(
                              fontSize: 13,
                              fontWeight: FontWeight.w600,
                              color: sel ? Colors.white : AppTheme.textPrimary,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 16),

          // Occupation type
          _label('I am a', Icons.work_rounded),
          const SizedBox(height: 6),
          Row(
            children: ['Student', 'Working Professional'].map((o) {
              final sel = _sOccupation == o;
              const color = AppTheme.teal;
              return Expanded(
                child: Padding(
                  padding: EdgeInsets.only(right: o == 'Student' ? 8 : 0),
                  child: GestureDetector(
                    onTap: () => setState(() => _sOccupation = o),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 160),
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      decoration: BoxDecoration(
                        color: sel ? color : AppTheme.card,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: sel ? color : AppTheme.border,
                          width: 0.5,
                        ),
                      ),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            o == 'Student'
                                ? Icons.school_rounded
                                : Icons.work_rounded,
                            size: 16,
                            color: sel ? Colors.white : AppTheme.textMuted,
                          ),
                          const SizedBox(width: 6),
                          Text(
                            o == 'Student' ? 'Student' : 'Professional',
                            style: GoogleFonts.dmSans(
                              fontSize: 12,
                              fontWeight: FontWeight.w600,
                              color: sel ? Colors.white : AppTheme.textPrimary,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 16),

          if (_sOccupation == 'Student') ...[
            _sectionHeader(
              'Academic Information',
              Icons.school_rounded,
              AppTheme.teal,
            ),
            const SizedBox(height: 10),
            _field(
              _sUniCtrl,
              'University / Institution',
              Icons.account_balance_rounded,
            ),
            const SizedBox(height: 10),
            // Education level
            _label('Education Level', Icons.military_tech_rounded),
            const SizedBox(height: 6),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: ['School', 'College', 'Bachelor', 'Master', 'PhD'].map((
                d,
              ) {
                final sel = _sDegree == d;
                return GestureDetector(
                  onTap: () => setState(() => _sDegree = d),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 160),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 9,
                    ),
                    decoration: BoxDecoration(
                      color: sel ? AppTheme.teal : AppTheme.card,
                      borderRadius: BorderRadius.circular(10),
                      border: Border.all(
                        color: sel ? AppTheme.teal : AppTheme.border,
                        width: 0.5,
                      ),
                    ),
                    child: Text(
                      d,
                      style: GoogleFonts.dmSans(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: sel ? Colors.white : AppTheme.textPrimary,
                      ),
                    ),
                  ),
                );
              }).toList(),
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                Expanded(
                  child: _field(
                    _sDeptCtrl,
                    'Department',
                    Icons.category_rounded,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _field(
                    _sSemCtrl,
                    'Semester / Year',
                    Icons.timeline_rounded,
                    type: TextInputType.number,
                  ),
                ),
              ],
            ),
          ] else ...[
            _sectionHeader(
              'Job Information',
              Icons.work_rounded,
              AppTheme.teal,
            ),
            const SizedBox(height: 10),
            _field(_sUniCtrl, 'Company / Organization', Icons.business_rounded),
            const SizedBox(height: 10),
            Row(
              children: [
                Expanded(
                  child: _field(
                    _sDeptCtrl,
                    'Job Title / Role',
                    Icons.badge_rounded,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _field(
                    _sSemCtrl,
                    'Years of Experience',
                    Icons.timeline_rounded,
                    type: TextInputType.number,
                  ),
                ),
              ],
            ),
          ],
          const SizedBox(height: 16),

          _infoBox(
            'Your academic info helps match you with the right hostel and roommates.',
            AppTheme.teal,
          ),
          const SizedBox(height: 8),
        ],
      ),
      ),
    );
  }

  // ── Parent Tab ────────────────────────────────────────────────────────────
  Widget _buildParentTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      child: _formSurface(
        child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (_fieldErrors.isNotEmpty) ...[
            _validationBanner('Please correct the highlighted fields before continuing.'),
            const SizedBox(height: 12),
          ],
          // ── Father / Guardian ──────────────────────────────────────
          _sectionHeader(
            'Father / Guardian',
            Icons.man_rounded,
            AppTheme.teal,
          ),
          const SizedBox(height: 10),
          _field(_pNameCtrl, 'Full Name *', Icons.person_outline_rounded),
          const SizedBox(height: 10),
          Row(
            children: [
              Expanded(
                child: _field(
                  _pPhoneCtrl,
                  'Phone Number *',
                  Icons.phone_rounded,
                  type: TextInputType.phone,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: _field(
                  _pCnicCtrl,
                  'CNIC Number',
                  Icons.credit_card_rounded,
                  type: TextInputType.number,
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          // Father relationship chips
          _label("Relationship", Icons.people_rounded),
          const SizedBox(height: 6),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children:
                [
                  "Father",
                  "Uncle",
                  "Brother",
                  "Grandfather",
                  "Guardian",
                  "Other",
                ].map((r) {
                  final sel = _pRelation == r;
                  return GestureDetector(
                    onTap: () => setState(() => _pRelation = r),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 160),
                      padding: const EdgeInsets.symmetric(
                        horizontal: 14,
                        vertical: 8,
                      ),
                      decoration: BoxDecoration(
                        color: sel ? AppTheme.teal : AppTheme.card,
                        borderRadius: BorderRadius.circular(10),
                        border: Border.all(
                          color: sel ? AppTheme.teal : AppTheme.border,
                          width: 0.5,
                        ),
                      ),
                      child: Text(
                        r,
                        style: GoogleFonts.dmSans(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: sel ? Colors.white : AppTheme.textPrimary,
                        ),
                      ),
                    ),
                  );
                }).toList(),
          ),
          const SizedBox(height: 10),
          _field(
            _pEmailCtrl,
            'Email (optional)',
            Icons.email_outlined,
            type: TextInputType.emailAddress,
          ),
          const SizedBox(height: 10),
          _field(_pCityCtrl, 'City / Address', Icons.location_city_rounded),
          const SizedBox(height: 20),

          // ── Second Guardian (optional) ────────────────────────────
          if (!_addMother)
            GestureDetector(
              onTap: () => setState(() => _addMother = true),
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 13),
                decoration: BoxDecoration(
                  color: AppTheme.card,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(
                    color: AppTheme.teal.withOpacity(0.42),
                    width: 1,
                  ),
                ),
                child: Row(
                  children: [
                    Container(
                      width: 28,
                      height: 28,
                      decoration: BoxDecoration(
                        color: AppTheme.teal.withOpacity(0.12),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.add_rounded,
                        size: 16,
                        color: AppTheme.teal,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: FittedBox(
                        fit: BoxFit.scaleDown,
                        alignment: Alignment.centerLeft,
                        child: Text(
                          "Add Another Guardian / Parent",
                          maxLines: 1,
                          style: GoogleFonts.dmSans(
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                            color: AppTheme.teal,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            )
          else ...[
            Row(
              children: [
                _sectionHeader(
                  "Second Guardian / Parent",
                  Icons.people_alt_rounded,
                  AppTheme.teal,
                ),
                const Spacer(),
                GestureDetector(
                  onTap: () => setState(() {
                    _addMother = false;
                    _fieldErrors.remove(_mNameCtrl);
                    _fieldErrors.remove(_mPhoneCtrl);
                    _fieldErrors.remove(_mCnicCtrl);
                    _fieldErrors.remove(_mEmailCtrl);
                  }),
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 10,
                      vertical: 5,
                    ),
                    decoration: BoxDecoration(
                      color: Colors.red.shade50,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      'Remove',
                      style: GoogleFonts.dmSans(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        color: Colors.red.shade600,
                      ),
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            _field(_mNameCtrl, "Full Name *", Icons.person_outline_rounded),
            const SizedBox(height: 10),
            // Relationship dropdown style chips
            _label("Relationship", Icons.people_rounded),
            const SizedBox(height: 6),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children:
                  [
                    "Mother",
                    "Uncle",
                    "Aunt",
                    "Brother",
                    "Sister",
                    "Guardian",
                    "Other",
                  ].map((r) {
                    final sel = _mRelation == r;
                    return GestureDetector(
                      onTap: () => setState(() => _mRelation = r),
                      child: AnimatedContainer(
                        duration: const Duration(milliseconds: 160),
                        padding: const EdgeInsets.symmetric(
                          horizontal: 14,
                          vertical: 8,
                        ),
                        decoration: BoxDecoration(
                          color: sel ? AppTheme.teal : AppTheme.card,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                            color: sel ? AppTheme.teal : AppTheme.border,
                            width: 0.5,
                          ),
                        ),
                        child: Text(
                          r,
                          style: GoogleFonts.dmSans(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: sel ? Colors.white : AppTheme.textPrimary,
                          ),
                        ),
                      ),
                    );
                  }).toList(),
            ),
            const SizedBox(height: 10),
            Row(
              children: [
                Expanded(
                  child: _field(
                    _mPhoneCtrl,
                    "Phone Number *",
                    Icons.phone_rounded,
                    type: TextInputType.phone,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _field(
                    _mCnicCtrl,
                    "CNIC Number",
                    Icons.credit_card_rounded,
                    type: TextInputType.number,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),
            _field(
              _mEmailCtrl,
              "Email (optional)",
              Icons.email_outlined,
              type: TextInputType.emailAddress,
            ),
          ],
          const SizedBox(height: 14),

          _infoBox(
            'Parent information is kept confidential and used only for '
            'emergency contact and hostel communication.',
            AppTheme.teal,
          ),
          const SizedBox(height: 8),
        ],
      ),
      ),
    );
  }

  // ── Bottom bar ────────────────────────────────────────────────────────────
  Widget _buildBottomBar() {
    return Container(
      padding: EdgeInsets.fromLTRB(
        16,
        12,
        16,
        12 + MediaQuery.of(context).padding.bottom,
      ),
      decoration: BoxDecoration(
        color: AppTheme.card,
        border: Border(top: BorderSide(color: AppTheme.border, width: 0.5)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 10,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        children: [
          if (_tab == 1) ...[
            GestureDetector(
              onTap: () {
                setState(() => _tab = 0);
                _pageCtrl.animateToPage(
                  0,
                  duration: const Duration(milliseconds: 300),
                  curve: Curves.easeInOut,
                );
              },
              child: Container(
                width: 48,
                height: 52,
                decoration: BoxDecoration(
                  color: AppTheme.bgSecondary,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppTheme.border, width: 0.5),
                ),
                child: const Icon(
                  Icons.arrow_back_rounded,
                  color: AppTheme.textSecondary,
                  size: 20,
                ),
              ),
            ),
            const SizedBox(width: 10),
          ],
          Expanded(
            child: SizedBox(
              height: 52,
              child: ElevatedButton(
                onPressed: _tab == 0 ? _goToParentTab : _proceedToBooking,
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.teal,
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Text(
                      _tab == 0 ? 'Next: Parent Info' : 'Proceed to Booking',
                      style: GoogleFonts.dmSans(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Container(
                      width: 26,
                      height: 26,
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.2),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(
                        _tab == 0
                            ? Icons.arrow_forward_rounded
                            : Icons.hotel_rounded,
                        size: 14,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  Widget _validationBanner(String message) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: Colors.red.shade50,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.red.shade200),
      ),
      child: Row(
        children: [
          Icon(Icons.error_outline_rounded, size: 18, color: Colors.red.shade700),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              message,
              style: GoogleFonts.dmSans(
                fontSize: 11.5,
                fontWeight: FontWeight.w600,
                color: Colors.red.shade800,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _formSurface({required Widget child}) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.fromLTRB(14, 16, 14, 18),
      decoration: BoxDecoration(
        color: AppTheme.card.withOpacity(0.97),
        borderRadius: BorderRadius.circular(24),
        border: Border.all(color: AppTheme.teal.withOpacity(0.24)),
        boxShadow: [
          BoxShadow(
            color: AppTheme.teal.withOpacity(0.10),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: child,
    );
  }

  Widget _sectionHeader(String text, IconData icon, Color color) {
    return Row(
      children: [
        Container(
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.12),
            borderRadius: BorderRadius.circular(11),
            border: Border.all(color: AppTheme.teal.withOpacity(0.16)),
          ),
          child: Icon(icon, size: 18, color: AppTheme.teal),
        ),
        const SizedBox(width: 10),
        Text(
          text,
          style: GoogleFonts.dmSans(
            fontSize: 14,
            fontWeight: FontWeight.w700,
            color: AppTheme.textPrimary,
          ),
        ),
      ],
    );
  }

  Widget _label(String text, IconData icon) => Row(
    children: [
      Icon(icon, size: 13, color: AppTheme.teal),
      const SizedBox(width: 6),
      Text(
        text,
        style: GoogleFonts.dmSans(
          fontSize: 12,
          fontWeight: FontWeight.w600,
          color: AppTheme.textPrimary,
        ),
      ),
    ],
  );

  Widget _field(
    TextEditingController ctrl,
    String hint,
    IconData icon, {
    TextInputType? type,
    int maxLines = 1,
  }) {
    final error = _fieldErrors[ctrl];
    final lowerHint = hint.toLowerCase();
    final isCnic = lowerHint.contains('cnic');
    final isPhone = lowerHint.contains('phone');
    final isNumericDetail =
        lowerHint.contains('semester') || lowerHint.contains('experience');
    final formatters = <TextInputFormatter>[
      if (isCnic)
        TextInputFormatter.withFunction(_limitCnicToThirteenDigits),
      if (isPhone || isNumericDetail) FilteringTextInputFormatter.digitsOnly,
      LengthLimitingTextInputFormatter(
        isCnic
            ? 13
            : isPhone
            ? 15
            : isNumericDetail
            ? 2
            : 120,
      ),
    ];
    return TextField(
      controller: ctrl,
      keyboardType: type,
      maxLines: maxLines,
      inputFormatters: formatters,
      textCapitalization: lowerHint.contains('email')
          ? TextCapitalization.none
          : TextCapitalization.words,
      onChanged: (_) => _validateLive(ctrl, hint),
      style: GoogleFonts.dmSans(fontSize: 14, color: AppTheme.textPrimary),
      decoration: InputDecoration(
        filled: true,
        fillColor: AppTheme.card,
        hintText: hint,
        hintStyle: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13),
        prefixIcon: Container(
          margin: const EdgeInsets.all(7),
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.10),
            borderRadius: BorderRadius.circular(9),
          ),
          child: Icon(icon, size: 16, color: AppTheme.teal),
        ),
        errorText: error,
        errorMaxLines: 2,
        counterText: '',
        errorStyle: GoogleFonts.dmSans(
          fontSize: 10,
          color: Colors.red.shade700,
        ),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(15),
          borderSide: BorderSide(
            color: error == null ? AppTheme.border : Colors.red.shade400,
          ),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(15),
          borderSide: BorderSide(
            color: error == null ? AppTheme.border : Colors.red.shade400,
          ),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(15),
          borderSide: BorderSide(
            color: error == null ? AppTheme.teal : Colors.red.shade600,
            width: 1.4,
          ),
        ),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 14,
          vertical: 13,
        ),
      ),
    );
  }

  Widget _notifRow(
    String label,
    IconData icon,
    bool value,
    ValueChanged<bool> onChange,
  ) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: AppTheme.card,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppTheme.border, width: 0.5),
        ),
        child: Row(
          children: [
            Icon(icon, size: 16, color: AppTheme.textMuted),
            const SizedBox(width: 10),
            Expanded(
              child: Text(
                label,
                style: GoogleFonts.dmSans(
                  fontSize: 12,
                  color: AppTheme.textPrimary,
                ),
              ),
            ),
            Switch(
              value: value,
              onChanged: onChange,
              activeColor: AppTheme.teal,
              materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
            ),
          ],
        ),
      ),
    );
  }

  Widget _infoBox(String text, Color color) {
    return Container(
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: color.withOpacity(0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border(left: BorderSide(color: color, width: 3)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline_rounded, color: color, size: 15),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: GoogleFonts.dmSans(
                fontSize: 12,
                color: color,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }
}
