import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import 'booking_choice_screen.dart';

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
  final _sNameCtrl  = TextEditingController();
  final _sEmailCtrl = TextEditingController();
  final _sPhoneCtrl = TextEditingController();
  final _sCnicCtrl  = TextEditingController();
  final _sUniCtrl   = TextEditingController();
  final _sDeptCtrl  = TextEditingController();
  final _sSemCtrl   = TextEditingController();
  String _sGender      = 'Female';
  String _sOccupation  = 'Student'; // Student or Working Professional
  String _sDegree      = 'Bachelor'; // Bachelor or Master

  // Father / Guardian info
  final _pNameCtrl  = TextEditingController();
  final _pPhoneCtrl = TextEditingController();
  final _pCnicCtrl  = TextEditingController();
  String _pRelation = 'Father';
  final _pCityCtrl  = TextEditingController();
  final _pEmailCtrl = TextEditingController();

  // Mother info (optional)
  bool _addMother = false;
  String _mRelation = 'Mother';
  final _mNameCtrl  = TextEditingController();
  final _mPhoneCtrl = TextEditingController();
  final _mCnicCtrl  = TextEditingController();
  final _mEmailCtrl = TextEditingController();

  late final AnimationController _anim;
  late final Animation<double> _fade;

  @override
  void initState() {
    super.initState();
    _anim = AnimationController(vsync: this,
        duration: const Duration(milliseconds: 400));
    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);
    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    _pageCtrl.dispose();
    for (final c in [
      _sNameCtrl, _sEmailCtrl, _sPhoneCtrl, _sCnicCtrl,
      _sUniCtrl, _sDeptCtrl, _sSemCtrl,
      _pNameCtrl, _pPhoneCtrl, _pCnicCtrl, _pCityCtrl, _pEmailCtrl,
      _mNameCtrl, _mPhoneCtrl, _mCnicCtrl, _mEmailCtrl,
    ]) { c.dispose(); }
    super.dispose();
  }

  void _goToParentTab() {
    if (_sNameCtrl.text.trim().isEmpty || _sPhoneCtrl.text.trim().isEmpty) {
      _showError('Please fill student name and phone number');
      return;
    }
    setState(() => _tab = 1);
    _pageCtrl.animateToPage(1,
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut);
  }

  void _proceedToBooking() {
    if (_pNameCtrl.text.trim().isEmpty || _pPhoneCtrl.text.trim().isEmpty) {
      _showError('Please fill parent name and phone number');
      return;
    }
    Navigator.push(context, MaterialPageRoute(
      builder: (_) => BookingChoiceScreen(
        hostelId: widget.hostelId,
        hostelName: widget.hostelName,
        city: widget.city,
        address: widget.address,
        studentName: _sNameCtrl.text.trim(),
        studentPhone: _sPhoneCtrl.text.trim(),
        studentEmail: _sEmailCtrl.text.trim(),
        studentCnic: _sCnicCtrl.text.trim(),
        studentUniversity: _sUniCtrl.text.trim(),
        parentName: _pNameCtrl.text.trim(),
        parentPhone: _pPhoneCtrl.text.trim(),
        parentRelation: _pRelation,
      ),
    ));
  }

  void _showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(SnackBar(
      content: Text(msg, style: GoogleFonts.dmSans()),
      backgroundColor: Colors.red.shade600,
      behavior: SnackBarBehavior.floating,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      margin: const EdgeInsets.all(16)));
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        body: Column(children: [
          _buildHeader(),
          // Tab bar
          _buildTabBar(),
          Expanded(
            child: PageView(
              controller: _pageCtrl,
              physics: const NeverScrollableScrollPhysics(),
              children: [
                _buildStudentTab(),
                _buildParentTab(),
              ],
            ),
          ),
          _buildBottomBar(),
        ]),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────
  Widget _buildHeader() {
    final isGirls = widget.hostelType.toLowerCase().contains('girl');
    final typeColor = isGirls
        ? const Color(0xFFE91E8C)
        : const Color(0xFF185FA5);

    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft, end: Alignment.bottomRight,
          colors: [AppTheme.tealDeep, AppTheme.teal]),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
          child: Column(children: [
            Row(children: [
              GestureDetector(
                onTap: () => Navigator.pop(context),
                child: Container(
                  width: 38, height: 38,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    shape: BoxShape.circle,
                    border: Border.all(
                        color: Colors.white.withOpacity(0.3))),
                  child: const Icon(Icons.arrow_back_rounded,
                      color: Colors.white, size: 18)),
              ),
              const SizedBox(width: 14),
              Expanded(child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                Text('Booking Information',
                    style: GoogleFonts.playfairDisplay(
                        fontSize: 20, fontWeight: FontWeight.w700,
                        color: Colors.white)),
                Text('Fill your details to proceed',
                    style: GoogleFonts.dmSans(
                        fontSize: 11, color: Colors.white60)),
              ])),
            ]),
            const SizedBox(height: 14),

            // Hostel info bar
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.white.withOpacity(0.1),
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                    color: Colors.white.withOpacity(0.2))),
              child: Row(children: [
                Container(
                  width: 40, height: 40,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(10)),
                  child: const Icon(Icons.apartment_rounded,
                      color: Colors.white, size: 20)),
                const SizedBox(width: 10),
                Expanded(child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                  Text(widget.hostelName,
                      style: GoogleFonts.dmSans(
                          fontSize: 13, fontWeight: FontWeight.w700,
                          color: Colors.white),
                      maxLines: 1, overflow: TextOverflow.ellipsis),
                  Text('${widget.city}  ·  ${widget.distanceKm.toStringAsFixed(1)} km',
                      style: GoogleFonts.dmSans(
                          fontSize: 11, color: Colors.white60)),
                ])),
                if (widget.pricePerMonth > 0)
                  Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 10, vertical: 5),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.15),
                      borderRadius: BorderRadius.circular(8)),
                    child: Text(
                      'PKR ${widget.pricePerMonth}/mo',
                      style: GoogleFonts.dmSans(
                          fontSize: 11, fontWeight: FontWeight.w700,
                          color: Colors.white)),
                  ),
              ]),
            ),
          ]),
        ),
      ),
    );
  }

  // ── Tab bar ───────────────────────────────────────────────────────────────
  Widget _buildTabBar() {
    return Container(
      color: AppTheme.bg,
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      child: Row(children: [
        Expanded(child: _tabBtn(0, 'Student Info',
            Icons.school_rounded, AppTheme.teal)),
        const SizedBox(width: 10),
        Expanded(child: _tabBtn(1, 'Parent Info',
            Icons.family_restroom_rounded, const Color(0xFF7B3FC4))),
      ]),
    );
  }

  Widget _tabBtn(int idx, String label, IconData icon, Color color) {
    final sel = _tab == idx;
    return GestureDetector(
      onTap: idx == 1 ? _goToParentTab
          : () {
              setState(() => _tab = 0);
              _pageCtrl.animateToPage(0,
                  duration: const Duration(milliseconds: 300),
                  curve: Curves.easeInOut);
            },
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 200),
        padding: const EdgeInsets.symmetric(vertical: 10),
        decoration: BoxDecoration(
          color: sel ? color : AppTheme.card,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
              color: sel ? color : AppTheme.border, width: 0.5)),
        child: Row(mainAxisAlignment: MainAxisAlignment.center,
            children: [
          Icon(icon, size: 16,
              color: sel ? Colors.white : AppTheme.textMuted),
          const SizedBox(width: 7),
          Text(label, style: GoogleFonts.dmSans(
              fontSize: 13, fontWeight: FontWeight.w600,
              color: sel ? Colors.white : AppTheme.textMuted)),
          if (sel) ...[
            const SizedBox(width: 6),
            Container(
              width: 6, height: 6,
              decoration: const BoxDecoration(
                  color: Colors.white, shape: BoxShape.circle)),
          ],
        ]),
      ),
    );
  }

  // ── Student Tab ───────────────────────────────────────────────────────────
  Widget _buildStudentTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start,
          children: [
        _sectionHeader('Personal Information',
            Icons.person_rounded, AppTheme.teal),
        const SizedBox(height: 10),
        _field(_sNameCtrl, 'Full Name *', Icons.person_outline_rounded),
        const SizedBox(height: 10),
        Row(children: [
          Expanded(child: _field(
              _sEmailCtrl, 'Email Address',
              Icons.email_outlined,
              type: TextInputType.emailAddress)),
          const SizedBox(width: 10),
          Expanded(child: _field(
              _sPhoneCtrl, 'Phone Number *',
              Icons.phone_rounded,
              type: TextInputType.phone)),
        ]),
        const SizedBox(height: 10),
        _field(_sCnicCtrl, 'CNIC / B-Form Number',
            Icons.credit_card_rounded,
            type: TextInputType.number),
        const SizedBox(height: 10),

        // Gender selector
        _label('Gender', Icons.wc_rounded),
        const SizedBox(height: 6),
        Row(children: ['Female', 'Male'].map((g) {
          final sel = _sGender == g;
          final color = g == 'Female'
              ? const Color(0xFFE91E8C)
              : const Color(0xFF185FA5);
          return Expanded(child: Padding(
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
                      width: 0.5)),
                child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                  Icon(
                    g == 'Female'
                        ? Icons.female_rounded
                        : Icons.male_rounded,
                    size: 18,
                    color: sel ? Colors.white : AppTheme.textMuted),
                  const SizedBox(width: 6),
                  Text(g, style: GoogleFonts.dmSans(
                      fontSize: 13, fontWeight: FontWeight.w600,
                      color: sel
                          ? Colors.white
                          : AppTheme.textPrimary)),
                ]),
              ),
            ),
          ));
        }).toList()),
        const SizedBox(height: 16),

        // Occupation type
        _label('I am a', Icons.work_rounded),
        const SizedBox(height: 6),
        Row(children: ['Student', 'Working Professional'].map((o) {
          final sel = _sOccupation == o;
          final color = o == 'Student'
              ? AppTheme.teal
              : const Color(0xFF7B3FC4);
          return Expanded(child: Padding(
            padding: EdgeInsets.only(right: o == 'Student' ? 8 : 0),
            child: GestureDetector(
              onTap: () => setState(() => _sOccupation = o),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 160),
                padding: const EdgeInsets.symmetric(vertical: 12),
                decoration: BoxDecoration(
                  color: sel ? color : AppTheme.card,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: sel ? color : AppTheme.border, width: 0.5)),
                child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                  Icon(o == 'Student' ? Icons.school_rounded : Icons.work_rounded,
                      size: 16, color: sel ? Colors.white : AppTheme.textMuted),
                  const SizedBox(width: 6),
                  Text(o == 'Student' ? 'Student' : 'Professional',
                      style: GoogleFonts.dmSans(fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: sel ? Colors.white : AppTheme.textPrimary)),
                ]),
              ),
            ),
          ));
        }).toList()),
        const SizedBox(height: 16),

        if (_sOccupation == 'Student') ...[
          _sectionHeader('Academic Information',
              Icons.school_rounded, const Color(0xFF185FA5)),
          const SizedBox(height: 10),
          _field(_sUniCtrl, 'University / Institution',
              Icons.account_balance_rounded),
          const SizedBox(height: 10),
          // Education level
          _label('Education Level', Icons.military_tech_rounded),
          const SizedBox(height: 6),
          Wrap(spacing: 8, runSpacing: 8,
            children: ['School', 'College', 'Bachelor', 'Master', 'PhD'].map((d) {
              final sel = _sDegree == d;
              return GestureDetector(
                onTap: () => setState(() => _sDegree = d),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 160),
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 9),
                  decoration: BoxDecoration(
                    color: sel ? const Color(0xFF185FA5) : AppTheme.card,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                        color: sel ? const Color(0xFF185FA5) : AppTheme.border,
                        width: 0.5)),
                  child: Text(d,
                      style: GoogleFonts.dmSans(fontSize: 13,
                          fontWeight: FontWeight.w600,
                          color: sel ? Colors.white : AppTheme.textPrimary)),
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 10),
          Row(children: [
            Expanded(child: _field(
                _sDeptCtrl, 'Department',
                Icons.category_rounded)),
            const SizedBox(width: 10),
            Expanded(child: _field(
                _sSemCtrl, 'Semester / Year',
                Icons.timeline_rounded,
                type: TextInputType.number)),
          ]),
        ] else ...[
          _sectionHeader('Job Information',
              Icons.work_rounded, const Color(0xFF7B3FC4)),
          const SizedBox(height: 10),
          _field(_sUniCtrl, 'Company / Organization',
              Icons.business_rounded),
          const SizedBox(height: 10),
          Row(children: [
            Expanded(child: _field(
                _sDeptCtrl, 'Job Title / Role',
                Icons.badge_rounded)),
            const SizedBox(width: 10),
            Expanded(child: _field(
                _sSemCtrl, 'Years of Experience',
                Icons.timeline_rounded,
                type: TextInputType.number)),
          ]),
        ],
        const SizedBox(height: 16),

        _infoBox('Your academic info helps match you with the right hostel and roommates.',
            AppTheme.teal),
        const SizedBox(height: 20),
      ]),
    );
  }

  // ── Parent Tab ────────────────────────────────────────────────────────────
  Widget _buildParentTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 14, 16, 16),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start,
          children: [
        // ── Father / Guardian ──────────────────────────────────────
        _sectionHeader('Father / Guardian',
            Icons.man_rounded, const Color(0xFF7B3FC4)),
        const SizedBox(height: 10),
        _field(_pNameCtrl, 'Full Name *',
            Icons.person_outline_rounded),
        const SizedBox(height: 10),
        Row(children: [
          Expanded(child: _field(
              _pPhoneCtrl, 'Phone Number *',
              Icons.phone_rounded,
              type: TextInputType.phone)),
          const SizedBox(width: 10),
          Expanded(child: _field(
              _pCnicCtrl, 'CNIC Number',
              Icons.credit_card_rounded,
              type: TextInputType.number)),
        ]),
        const SizedBox(height: 10),
        // Father relationship chips
        _label("Relationship", Icons.people_rounded),
        const SizedBox(height: 6),
        Wrap(spacing: 8, runSpacing: 8,
          children: ["Father", "Uncle", "Brother",
                     "Grandfather", "Guardian", "Other"].map((r) {
            final sel = _pRelation == r;
            return GestureDetector(
              onTap: () => setState(() => _pRelation = r),
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 160),
                padding: const EdgeInsets.symmetric(
                    horizontal: 14, vertical: 8),
                decoration: BoxDecoration(
                  color: sel
                      ? const Color(0xFF7B3FC4)
                      : AppTheme.card,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                      color: sel
                          ? const Color(0xFF7B3FC4)
                          : AppTheme.border,
                      width: 0.5)),
                child: Text(r, style: GoogleFonts.dmSans(
                    fontSize: 12, fontWeight: FontWeight.w600,
                    color: sel
                        ? Colors.white
                        : AppTheme.textPrimary)),
              ),
            );
          }).toList(),
        ),
        const SizedBox(height: 10),
        _field(_pEmailCtrl, 'Email (optional)',
            Icons.email_outlined,
            type: TextInputType.emailAddress),
        const SizedBox(height: 10),
        _field(_pCityCtrl, 'City / Address',
            Icons.location_city_rounded),
        const SizedBox(height: 20),

        // ── Second Guardian (optional) ────────────────────────────
        if (!_addMother)
          GestureDetector(
            onTap: () => setState(() => _addMother = true),
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 13),
              decoration: BoxDecoration(
                color: AppTheme.card,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                    color: const Color(0xFF7B3FC4).withOpacity(0.4),
                    width: 1),
              ),
              child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                Container(
                  width: 28, height: 28,
                  decoration: BoxDecoration(
                    color: const Color(0xFF7B3FC4).withOpacity(0.1),
                    shape: BoxShape.circle),
                  child: const Icon(Icons.add_rounded,
                      size: 16, color: Color(0xFF7B3FC4))),
                const SizedBox(width: 10),
                Text("Add Another Guardian / Parent",
                    style: GoogleFonts.dmSans(
                        fontSize: 14, fontWeight: FontWeight.w600,
                        color: const Color(0xFF7B3FC4))),
              ]),
            ),
          )
        else ...[
          Row(children: [
            _sectionHeader("Second Guardian / Parent",
                Icons.people_alt_rounded, const Color(0xFFE91E8C)),
            const Spacer(),
            GestureDetector(
              onTap: () => setState(() => _addMother = false),
              child: Container(
                padding: const EdgeInsets.symmetric(
                    horizontal: 10, vertical: 5),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8)),
                child: Text('Remove',
                    style: GoogleFonts.dmSans(
                        fontSize: 11, fontWeight: FontWeight.w600,
                        color: Colors.red.shade600)),
              ),
            ),
          ]),
          const SizedBox(height: 10),
          _field(_mNameCtrl, "Full Name *",
              Icons.person_outline_rounded),
          const SizedBox(height: 10),
          // Relationship dropdown style chips
          _label("Relationship", Icons.people_rounded),
          const SizedBox(height: 6),
          Wrap(spacing: 8, runSpacing: 8,
            children: ["Mother", "Uncle", "Aunt", "Brother",
                       "Sister", "Guardian", "Other"].map((r) {
              final sel = _mRelation == r;
              return GestureDetector(
                onTap: () => setState(() => _mRelation = r),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 160),
                  padding: const EdgeInsets.symmetric(
                      horizontal: 14, vertical: 8),
                  decoration: BoxDecoration(
                    color: sel
                        ? const Color(0xFFE91E8C)
                        : AppTheme.card,
                    borderRadius: BorderRadius.circular(10),
                    border: Border.all(
                        color: sel
                            ? const Color(0xFFE91E8C)
                            : AppTheme.border,
                        width: 0.5)),
                  child: Text(r, style: GoogleFonts.dmSans(
                      fontSize: 12, fontWeight: FontWeight.w600,
                      color: sel
                          ? Colors.white
                          : AppTheme.textPrimary)),
                ),
              );
            }).toList(),
          ),
          const SizedBox(height: 10),
          Row(children: [
            Expanded(child: _field(
                _mPhoneCtrl, "Phone Number *",
                Icons.phone_rounded,
                type: TextInputType.phone)),
            const SizedBox(width: 10),
            Expanded(child: _field(
                _mCnicCtrl, "CNIC Number",
                Icons.credit_card_rounded,
                type: TextInputType.number)),
          ]),
          const SizedBox(height: 10),
          _field(_mEmailCtrl, "Email (optional)",
              Icons.email_outlined,
              type: TextInputType.emailAddress),
        ],
        const SizedBox(height: 14),

        _infoBox(
          'Parent information is kept confidential and used only for '
          'emergency contact and hostel communication.',
          const Color(0xFF7B3FC4)),
        const SizedBox(height: 20),
      ]),
    );
  }

  // ── Bottom bar ────────────────────────────────────────────────────────────
  Widget _buildBottomBar() {
    return Container(
      padding: EdgeInsets.fromLTRB(16, 12, 16,
          12 + MediaQuery.of(context).padding.bottom),
      decoration: BoxDecoration(
        color: AppTheme.card,
        border: Border(top: BorderSide(
            color: AppTheme.border, width: 0.5)),
        boxShadow: [BoxShadow(
            color: Colors.black.withOpacity(0.06),
            blurRadius: 10, offset: const Offset(0, -2))],
      ),
      child: Row(children: [
        if (_tab == 1) ...[
          GestureDetector(
            onTap: () {
              setState(() => _tab = 0);
              _pageCtrl.animateToPage(0,
                  duration: const Duration(milliseconds: 300),
                  curve: Curves.easeInOut);
            },
            child: Container(
              width: 48, height: 52,
              decoration: BoxDecoration(
                color: AppTheme.bgSecondary,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(
                    color: AppTheme.border, width: 0.5)),
              child: const Icon(Icons.arrow_back_rounded,
                  color: AppTheme.textSecondary, size: 20)),
          ),
          const SizedBox(width: 10),
        ],
        Expanded(
          child: SizedBox(
            height: 52,
            child: ElevatedButton(
              onPressed: _tab == 0 ? _goToParentTab : _proceedToBooking,
              style: ElevatedButton.styleFrom(
                backgroundColor: _tab == 0
                    ? AppTheme.teal
                    : const Color(0xFF7B3FC4),
                foregroundColor: Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14))),
              child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                Text(
                  _tab == 0
                      ? 'Next: Parent Info'
                      : 'Proceed to Booking',
                  style: GoogleFonts.dmSans(
                      fontSize: 15, fontWeight: FontWeight.w600)),
                const SizedBox(width: 10),
                Container(
                  width: 26, height: 26,
                  decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      shape: BoxShape.circle),
                  child: Icon(
                    _tab == 0
                        ? Icons.arrow_forward_rounded
                        : Icons.hotel_rounded,
                    size: 14)),
              ]),
            ),
          ),
        ),
      ]),
    );
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  Widget _sectionHeader(String text, IconData icon, Color color) {
    return Row(children: [
      Container(
        width: 32, height: 32,
        decoration: BoxDecoration(
          color: color.withOpacity(0.1), shape: BoxShape.circle),
        child: Icon(icon, size: 16, color: color)),
      const SizedBox(width: 10),
      Text(text, style: GoogleFonts.dmSans(
          fontSize: 14, fontWeight: FontWeight.w700,
          color: AppTheme.textPrimary)),
    ]);
  }

  Widget _label(String text, IconData icon) => Row(children: [
    Icon(icon, size: 13, color: AppTheme.teal),
    const SizedBox(width: 6),
    Text(text, style: GoogleFonts.dmSans(
        fontSize: 12, fontWeight: FontWeight.w600,
        color: AppTheme.textPrimary)),
  ]);

  Widget _field(TextEditingController ctrl, String hint, IconData icon,
      {TextInputType? type, int maxLines = 1}) {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.card, borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.border, width: 0.5)),
      child: TextField(
        controller: ctrl, keyboardType: type, maxLines: maxLines,
        style: GoogleFonts.dmSans(
            fontSize: 14, color: AppTheme.textPrimary),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(
              color: AppTheme.textMuted, fontSize: 13),
          prefixIcon: Icon(icon, size: 17, color: AppTheme.textMuted),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(
              horizontal: 14, vertical: 13)),
      ),
    );
  }

  Widget _notifRow(String label, IconData icon, bool value,
      ValueChanged<bool> onChange) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Container(
        padding: const EdgeInsets.symmetric(
            horizontal: 14, vertical: 10),
        decoration: BoxDecoration(
          color: AppTheme.card, borderRadius: BorderRadius.circular(12),
          border: Border.all(color: AppTheme.border, width: 0.5)),
        child: Row(children: [
          Icon(icon, size: 16, color: AppTheme.textMuted),
          const SizedBox(width: 10),
          Expanded(child: Text(label, style: GoogleFonts.dmSans(
              fontSize: 12, color: AppTheme.textPrimary))),
          Switch(value: value, onChanged: onChange,
              activeColor: const Color(0xFF7B3FC4),
              materialTapTargetSize: MaterialTapTargetSize.shrinkWrap),
        ]),
      ),
    );
  }

  Widget _infoBox(String text, Color color) {
    return Container(
      padding: const EdgeInsets.all(13),
      decoration: BoxDecoration(
        color: color.withOpacity(0.05),
        borderRadius: BorderRadius.circular(12),
        border: Border(left: BorderSide(color: color, width: 3))),
      child: Row(crossAxisAlignment: CrossAxisAlignment.start,
          children: [
        Icon(Icons.info_outline_rounded, color: color, size: 15),
        const SizedBox(width: 10),
        Expanded(child: Text(text, style: GoogleFonts.dmSans(
            fontSize: 12, color: color, height: 1.4))),
      ]),
    );
  }
}