import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'my_bookings_screen.dart';

class BookingScreen extends StatefulWidget {
  final int hostelId;
  final String hostelName;
  final String city;
  final String address;
  final String studentName;
  final String studentPhone;
  final String studentEmail;
  final String studentCnic;
  final String studentUniversity;
  final String studentDepartment;
  final String studentSemester;
  final String studentGender;
  final String studentOccupation;
  final String studentDegree;
  final String parentName;
  final String parentPhone;
  final String parentRelation;
  final String parentCnic;
  final String parentCity;
  final String parentEmail;
  final String secondaryGuardianName;
  final String secondaryGuardianPhone;
  final String secondaryGuardianCnic;
  final String secondaryGuardianRelation;
  final String secondaryGuardianEmail;

  const BookingScreen({
    super.key,
    required this.hostelId,
    required this.hostelName,
    required this.city,
    required this.address,
    this.studentName = '',
    this.studentPhone = '',
    this.studentEmail = '',
    this.studentCnic = '',
    this.studentUniversity = '',
    this.studentDepartment = '',
    this.studentSemester = '',
    this.studentGender = '',
    this.studentOccupation = '',
    this.studentDegree = '',
    this.parentName = '',
    this.parentPhone = '',
    this.parentRelation = '',
    this.parentCnic = '',
    this.parentCity = '',
    this.parentEmail = '',
    this.secondaryGuardianName = '',
    this.secondaryGuardianPhone = '',
    this.secondaryGuardianCnic = '',
    this.secondaryGuardianRelation = '',
    this.secondaryGuardianEmail = '',
  });

  @override
  State<BookingScreen> createState() => _BookingScreenState();
}

class _BookingScreenState extends State<BookingScreen>
    with SingleTickerProviderStateMixin {
  // Step: 0 = room select, 1 = details, 2 = confirm, 3 = success
  int _step = 0;

  // Step 1 selections
  String? _selectedRoom;
  int _duration = 1; // months
  bool _customDuration = false;
  final _customDurCtrl = TextEditingController();
  DateTime _checkIn = DateTime.now().add(const Duration(days: 1));

  // Step 2 details
  final _nameCtrl = TextEditingController();
  final _phoneCtrl = TextEditingController();
  final _cnicCtrl = TextEditingController();
  final _uniCtrl = TextEditingController();
  final _reqCtrl = TextEditingController();

  bool _submitting = false;
  String? _bookingId;
  final Map<TextEditingController, String> _fieldErrors = {};
  String? _durationError;

  late final AnimationController _anim;
  late final Animation<double> _fade;

  List<Map<String, dynamic>> _roomTypes = [];

  final _durations = [
    {'label': '1 Month', 'months': 1},
    {'label': '3 Months', 'months': 3},
    {'label': '6 Months', 'months': 6},
    {'label': '1 Year', 'months': 12},
  ];

  @override
  void initState() {
    super.initState();
    if (widget.studentName.isNotEmpty) _nameCtrl.text = widget.studentName;
    if (widget.studentPhone.isNotEmpty) _phoneCtrl.text = widget.studentPhone;
    if (widget.studentCnic.isNotEmpty) _cnicCtrl.text = widget.studentCnic;
    if (widget.studentUniversity.isNotEmpty)
      _uniCtrl.text = widget.studentUniversity;
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
    _customDurCtrl.dispose();
    _nameCtrl.dispose();
    _phoneCtrl.dispose();
    _cnicCtrl.dispose();
    _uniCtrl.dispose();
    _reqCtrl.dispose();
    super.dispose();
  }

  Map<String, dynamic>? get _room => _selectedRoom == null
      ? null
      : _roomTypes.firstWhere(
          (r) => r['type'] == _selectedRoom,
          orElse: () => _roomTypes.first,
        );

  int get _totalPrice => (_room?['price'] as int? ?? 0) * _duration;

  String? get _requestedRoomType {
    return null;
  }

  void _nextStep() {
    if (_step == 0 && _customDuration && _durationError != null) {
      ScaffoldMessenger.of(
        context,
      ).showSnackBar(_validationSnack('Please enter a valid stay duration'));
      return;
    }
    if (_step == 1) {
      final errors = <TextEditingController, String>{};
      final digits = _phoneCtrl.text.replaceAll(RegExp(r'[^0-9+]'), '');
      if (_nameCtrl.text.trim().length < 2) {
        errors[_nameCtrl] = 'Full name is required';
      }
      if (digits.length < 10 || digits.length > 15) {
        errors[_phoneCtrl] = 'Enter a valid phone number';
      }
      final cnicDigits = _cnicCtrl.text.replaceAll(RegExp(r'[^0-9]'), '');
      if (_cnicCtrl.text.trim().isNotEmpty && cnicDigits.length != 13) {
        errors[_cnicCtrl] = 'CNIC/B-Form must contain 13 digits';
      }
      if (_uniCtrl.text.trim().isNotEmpty && _uniCtrl.text.trim().length < 2) {
        errors[_uniCtrl] = 'Enter a valid institution';
      }
      if (_reqCtrl.text.length > 500) {
        errors[_reqCtrl] = 'Keep requirements under 500 characters';
      }
      setState(() {
        _fieldErrors
          ..clear()
          ..addAll(errors);
      });
      if (errors.isNotEmpty) {
        ScaffoldMessenger.of(context).showSnackBar(
          _validationSnack('Please correct the highlighted fields'),
        );
        return;
      }
    }
    _anim.reset();
    setState(() => _step++);
    _anim.forward();
  }

  SnackBar _validationSnack(String message) => SnackBar(
    content: Text(message, style: GoogleFonts.dmSans()),
    backgroundColor: Colors.red.shade600,
    behavior: SnackBarBehavior.floating,
    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
    margin: const EdgeInsets.all(16),
  );

  void _prevStep() {
    if (_step == 0) {
      Navigator.pop(context);
      return;
    }
    _anim.reset();
    setState(() => _step--);
    _anim.forward();
  }

  Future<void> _submitBooking() async {
    setState(() => _submitting = true);
    try {
      final result = await Api().createBooking(
        widget.hostelId,
        _checkIn.toIso8601String(),
        _checkIn.add(Duration(days: _duration * 30)).toIso8601String(),
        requestedRoomType: _requestedRoomType,
        application: {
          'student_name': _nameCtrl.text.trim(),
          'student_email': widget.studentEmail,
          'student_phone': _phoneCtrl.text.trim(),
          'student_cnic': _cnicCtrl.text.trim(),
          'university': _uniCtrl.text.trim(),
          'department': widget.studentDepartment,
          'semester': widget.studentSemester,
          'gender': widget.studentGender,
          'occupation': widget.studentOccupation,
          'degree': widget.studentDegree,
          'parent_name': widget.parentName,
          'parent_phone': widget.parentPhone,
          'parent_cnic': widget.parentCnic,
          'parent_relation': widget.parentRelation,
          'parent_city': widget.parentCity,
          'parent_email': widget.parentEmail,
          'secondary_guardian_name': widget.secondaryGuardianName,
          'secondary_guardian_phone': widget.secondaryGuardianPhone,
          'secondary_guardian_cnic': widget.secondaryGuardianCnic,
          'secondary_guardian_relation': widget.secondaryGuardianRelation,
          'secondary_guardian_email': widget.secondaryGuardianEmail,
          'special_requirements': _reqCtrl.text.trim(),
        },
      );
      if (!mounted) return;
      final booking = result['booking'] is Map
          ? Map<String, dynamic>.from(result['booking'] as Map)
          : <String, dynamic>{};
      setState(() {
        _submitting = false;
        _bookingId = booking['id']?.toString() ?? 'Pending';
        _step = 3;
      });
      _anim.reset();
      _anim.forward();
    } catch (error) {
      if (!mounted) return;
      setState(() => _submitting = false);
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('Booking submit nahi ho saki: $error'),
          backgroundColor: Colors.red.shade700,
        ),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.48,
          tintColor: AppTheme.tealDeep,
          tintOpacity: 0.20,
          cardMargin: const EdgeInsets.all(12),
          cardRadius: 30,
          cardColor: AppTheme.bg.withOpacity(0.97),
          child: Column(
            children: [
              _buildHeader(),
              Expanded(
                child: FadeTransition(
                  opacity: _fade,
                  child: _step == 0
                      ? _buildRoomStep()
                      : _step == 1
                      ? _buildDetailsStep()
                      : _step == 2
                      ? _buildConfirmStep()
                      : _buildSuccessStep(),
                ),
              ),
              if (_step < 3) _buildBottomBar(),
            ],
          ),
        ),
      ),
    );
  }

  // ── Header ────────────────────────────────────────────────────────────────
  Widget _buildHeader() {
    final stepLabels = ['Choose Room', 'Your Details', 'Confirm'];
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
          padding: const EdgeInsets.fromLTRB(16, 12, 16, 18),
          child: Column(
            children: [
              Row(
                children: [
                  GestureDetector(
                    onTap: _prevStep,
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
                          'Book Hostel',
                          style: GoogleFonts.playfairDisplay(
                            fontSize: 20,
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ),
                        ),
                        Text(
                          widget.hostelName,
                          style: GoogleFonts.dmSans(
                            fontSize: 12,
                            color: Colors.white60,
                          ),
                        ),
                      ],
                    ),
                  ),
                  if (_step < 3)
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 5,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.white.withOpacity(0.15),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        _step < stepLabels.length ? stepLabels[_step] : 'Done',
                        style: GoogleFonts.dmSans(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                ],
              ),
              if (_step < 3) ...[
                const SizedBox(height: 14),
                // Step progress bar
                Row(
                  children: List.generate(
                    3,
                    (i) => Expanded(
                      child: Container(
                        height: 3,
                        margin: EdgeInsets.only(right: i < 2 ? 6 : 0),
                        decoration: BoxDecoration(
                          color: i <= _step
                              ? AppTheme.tealMint
                              : Colors.white.withOpacity(0.25),
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ],
          ),
        ),
      ),
    );
  }

  // ── Step 0: Room Selection ─────────────────────────────────────────────────
  Widget _buildRoomStep() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 18, 16, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionLabel('Booking request', Icons.send_rounded),
          const SizedBox(height: 12),

          _roomState(),

          // Room type cards
          ...List.generate(_roomTypes.length, (i) {
            final r = _roomTypes[i];
            final sel = _selectedRoom == r['type'];
            final color = r['color'] as Color;
            final avail = r['avail'] as int;

            return GestureDetector(
              onTap: avail > 0
                  ? () => setState(() => _selectedRoom = r['type'] as String)
                  : null,
              child: AnimatedContainer(
                duration: const Duration(milliseconds: 180),
                margin: const EdgeInsets.only(bottom: 10),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: sel ? color.withOpacity(0.06) : AppTheme.card,
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(
                    color: sel ? color : AppTheme.border,
                    width: sel ? 2 : 0.5,
                  ),
                  boxShadow: sel
                      ? [
                          BoxShadow(
                            color: color.withOpacity(0.15),
                            blurRadius: 12,
                            offset: const Offset(0, 4),
                          ),
                        ]
                      : [],
                ),
                child: Row(
                  children: [
                    Container(
                      width: 52,
                      height: 52,
                      decoration: BoxDecoration(
                        color: sel
                            ? color.withOpacity(0.12)
                            : AppTheme.bgSecondary,
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Icon(
                        r['icon'] as IconData,
                        size: 26,
                        color: sel ? color : AppTheme.textMuted,
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Text(
                                r['type'] as String,
                                style: GoogleFonts.dmSans(
                                  fontSize: 16,
                                  fontWeight: FontWeight.w700,
                                  color: sel ? color : AppTheme.textPrimary,
                                ),
                              ),
                              const SizedBox(width: 8),
                              if (avail == 0)
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 8,
                                    vertical: 2,
                                  ),
                                  decoration: BoxDecoration(
                                    color: Colors.red.shade50,
                                    borderRadius: BorderRadius.circular(6),
                                  ),
                                  child: Text(
                                    'Full',
                                    style: GoogleFonts.dmSans(
                                      fontSize: 10,
                                      color: Colors.red.shade600,
                                      fontWeight: FontWeight.w600,
                                    ),
                                  ),
                                ),
                            ],
                          ),
                          Text(
                            r['desc'] as String,
                            style: GoogleFonts.dmSans(
                              fontSize: 12,
                              color: AppTheme.textMuted,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Row(
                            children: [
                              Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 8,
                                  vertical: 3,
                                ),
                                decoration: BoxDecoration(
                                  color: avail > 0
                                      ? AppTheme.greenLight
                                      : Colors.red.shade50,
                                  borderRadius: BorderRadius.circular(6),
                                ),
                                child: Text(
                                  '$avail beds free',
                                  style: GoogleFonts.dmSans(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w600,
                                    color: avail > 0
                                        ? AppTheme.green
                                        : Colors.red.shade600,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          'PKR ${r['price']}',
                          style: GoogleFonts.dmSans(
                            fontSize: 16,
                            fontWeight: FontWeight.w700,
                            color: sel ? color : AppTheme.textPrimary,
                          ),
                        ),
                        Text(
                          '/ month',
                          style: GoogleFonts.dmSans(
                            fontSize: 11,
                            color: AppTheme.textMuted,
                          ),
                        ),
                        if (sel) ...[
                          const SizedBox(height: 6),
                          Container(
                            width: 24,
                            height: 24,
                            decoration: BoxDecoration(
                              color: color,
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(
                              Icons.check_rounded,
                              color: Colors.white,
                              size: 14,
                            ),
                          ),
                        ],
                      ],
                    ),
                  ],
                ),
              ),
            );
          }),

          const SizedBox(height: 20),
          _sectionLabel('Stay Duration', Icons.calendar_month_rounded),
          const SizedBox(height: 12),

          // Duration chips
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              ..._durations.map((d) {
                final sel = !_customDuration && _duration == d['months'];
                return GestureDetector(
                  onTap: () => setState(() {
                    _customDuration = false;
                    _duration = d['months'] as int;
                  }),
                  child: AnimatedContainer(
                    duration: const Duration(milliseconds: 160),
                    padding: const EdgeInsets.symmetric(
                      horizontal: 18,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      color: sel ? AppTheme.teal : AppTheme.card,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: sel ? AppTheme.teal : AppTheme.border,
                        width: 0.5,
                      ),
                    ),
                    child: Text(
                      d['label'] as String,
                      style: GoogleFonts.dmSans(
                        fontSize: 13,
                        fontWeight: FontWeight.w600,
                        color: sel ? Colors.white : AppTheme.textPrimary,
                      ),
                    ),
                  ),
                );
              }),
              // Custom option
              GestureDetector(
                onTap: () => setState(() => _customDuration = true),
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 160),
                  padding: const EdgeInsets.symmetric(
                    horizontal: 18,
                    vertical: 10,
                  ),
                  decoration: BoxDecoration(
                    color: _customDuration ? AppTheme.teal : AppTheme.card,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: _customDuration ? AppTheme.teal : AppTheme.border,
                      width: 0.5,
                    ),
                  ),
                  child: Text(
                    'Custom',
                    style: GoogleFonts.dmSans(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: _customDuration
                          ? Colors.white
                          : AppTheme.textPrimary,
                    ),
                  ),
                ),
              ),
            ],
          ),
          // Custom duration input
          if (_customDuration) ...[
            const SizedBox(height: 10),
            Container(
              decoration: BoxDecoration(
                color: AppTheme.card,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppTheme.teal.withOpacity(0.4)),
              ),
              child: TextField(
                controller: _customDurCtrl,
                keyboardType: TextInputType.number,
                onChanged: (v) {
                  final n = int.tryParse(v);
                  setState(() {
                    if (n != null && n >= 1 && n <= 24) {
                      _duration = n;
                      _durationError = null;
                    } else {
                      _durationError = 'Enter 1 to 24 months';
                    }
                  });
                },
                style: GoogleFonts.dmSans(
                  fontSize: 14,
                  color: AppTheme.textPrimary,
                ),
                decoration: InputDecoration(
                  hintText: 'Enter number of months (e.g. 2, 5, 8)',
                  hintStyle: GoogleFonts.dmSans(
                    color: AppTheme.textMuted,
                    fontSize: 13,
                  ),
                  prefixIcon: const Icon(
                    Icons.edit_calendar_rounded,
                    size: 18,
                    color: AppTheme.teal,
                  ),
                  suffixText: 'months',
                  suffixStyle: GoogleFonts.dmSans(
                    color: AppTheme.textMuted,
                    fontSize: 12,
                  ),
                  errorText: _durationError,
                  errorStyle: GoogleFonts.dmSans(
                    fontSize: 10,
                    color: Colors.red.shade700,
                  ),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(14),
                    borderSide: BorderSide(
                      color: _durationError == null
                          ? AppTheme.border
                          : Colors.red.shade400,
                    ),
                  ),
                  enabledBorder: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(14),
                    borderSide: BorderSide(
                      color: _durationError == null
                          ? AppTheme.border
                          : Colors.red.shade400,
                    ),
                  ),
                  contentPadding: const EdgeInsets.symmetric(
                    horizontal: 14,
                    vertical: 14,
                  ),
                ),
              ),
            ),
          ],
          const SizedBox(height: 20),

          // Check-in date
          _sectionLabel('Check-in Date', Icons.event_rounded),
          const SizedBox(height: 10),
          GestureDetector(
            onTap: () async {
              final picked = await showDatePicker(
                context: context,
                initialDate: _checkIn,
                firstDate: DateTime.now(),
                lastDate: DateTime.now().add(const Duration(days: 365)),
                builder: (ctx, child) => Theme(
                  data: Theme.of(ctx).copyWith(
                    colorScheme: const ColorScheme.light(
                      primary: AppTheme.teal,
                      onPrimary: Colors.white,
                    ),
                  ),
                  child: child!,
                ),
              );
              if (picked != null) setState(() => _checkIn = picked);
            },
            child: Container(
              padding: const EdgeInsets.all(14),
              decoration: BoxDecoration(
                color: AppTheme.card,
                borderRadius: BorderRadius.circular(14),
                border: Border.all(color: AppTheme.teal.withOpacity(0.4)),
              ),
              child: Row(
                children: [
                  const Icon(
                    Icons.event_rounded,
                    color: AppTheme.teal,
                    size: 20,
                  ),
                  const SizedBox(width: 12),
                  Text(
                    '${_checkIn.day}/${_checkIn.month}/${_checkIn.year}',
                    style: GoogleFonts.dmSans(
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      color: AppTheme.textPrimary,
                    ),
                  ),
                  const Spacer(),
                  const Icon(
                    Icons.edit_calendar_rounded,
                    color: AppTheme.textMuted,
                    size: 16,
                  ),
                ],
              ),
            ),
          ),

          // Price preview
          if (_selectedRoom != null) ...[
            const SizedBox(height: 20),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    AppTheme.teal.withOpacity(0.08),
                    AppTheme.teal.withOpacity(0.03),
                  ],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppTheme.teal.withOpacity(0.2)),
              ),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        '${_room?['type']} Room × $_duration months',
                        style: GoogleFonts.dmSans(
                          fontSize: 13,
                          color: AppTheme.textMuted,
                        ),
                      ),
                      Text(
                        'PKR ${_room?['price']} × $_duration',
                        style: GoogleFonts.dmSans(
                          fontSize: 13,
                          color: AppTheme.textMuted,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Divider(height: 1, color: AppTheme.border, thickness: 0.5),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        'Total Amount',
                        style: GoogleFonts.dmSans(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          color: AppTheme.textPrimary,
                        ),
                      ),
                      Text(
                        'PKR ${_totalPrice.toString().replaceAllMapped(RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'), (m) => '${m[1]},')}',
                        style: GoogleFonts.dmSans(
                          fontSize: 18,
                          fontWeight: FontWeight.w700,
                          color: AppTheme.teal,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  // ── Step 1: Student Details ────────────────────────────────────────────────
  Widget _buildDetailsStep() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 18, 16, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Booking summary chip
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.06),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppTheme.teal.withOpacity(0.2)),
            ),
            child: Row(
              children: [
                const Icon(
                  Icons.meeting_room_rounded,
                  color: AppTheme.teal,
                  size: 18,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        '${_room?['type']} Room at ${widget.hostelName}',
                        style: GoogleFonts.dmSans(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          color: AppTheme.textPrimary,
                        ),
                      ),
                      Text(
                        '$_duration month${_duration > 1 ? 's' : ''}  ·  PKR $_totalPrice total',
                        style: GoogleFonts.dmSans(
                          fontSize: 11,
                          color: AppTheme.teal,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),

          _sectionLabel('Personal Information', Icons.person_rounded),
          const SizedBox(height: 10),
          _field(_nameCtrl, 'Full Name *', Icons.person_outline_rounded),
          const SizedBox(height: 10),
          _field(
            _phoneCtrl,
            'Phone Number *',
            Icons.phone_rounded,
            type: TextInputType.phone,
          ),
          const SizedBox(height: 10),
          _field(
            _cnicCtrl,
            'CNIC / B-Form Number',
            Icons.credit_card_rounded,
            type: TextInputType.number,
          ),
          const SizedBox(height: 10),
          _field(_uniCtrl, 'University / Institution', Icons.school_rounded),
          const SizedBox(height: 16),

          _sectionLabel('Special Requirements', Icons.notes_rounded),
          const SizedBox(height: 10),
          _field(
            _reqCtrl,
            'Any special requests or requirements...',
            Icons.notes_rounded,
            maxLines: 3,
          ),
          const SizedBox(height: 16),

          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.amber.shade50,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.amber.shade200, width: 0.5),
            ),
            child: Row(
              children: [
                Icon(
                  Icons.info_outline_rounded,
                  color: Colors.amber.shade700,
                  size: 16,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Your booking request will be sent to the hostel owner for approval.',
                    style: GoogleFonts.dmSans(
                      fontSize: 12,
                      color: Colors.amber.shade800,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  // ── Step 2: Confirm ────────────────────────────────────────────────────────
  Widget _buildConfirmStep() {
    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(16, 18, 16, 16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionLabel('Booking Summary', Icons.receipt_long_rounded),
          const SizedBox(height: 12),

          Container(
            decoration: BoxDecoration(
              color: AppTheme.card,
              borderRadius: BorderRadius.circular(18),
              border: Border.all(color: AppTheme.border, width: 0.5),
            ),
            child: Column(
              children: [
                // Hostel
                _confirmRow(
                  Icons.apartment_rounded,
                  'Hostel',
                  widget.hostelName,
                  AppTheme.teal,
                ),
                _cDivider(),
                _confirmRow(
                  Icons.location_on_rounded,
                  'Location',
                  widget.city,
                  AppTheme.textMuted,
                ),
                _cDivider(),
                _confirmRow(
                  Icons.meeting_room_rounded,
                  'Room Type',
                  '${_room?['type']} Room',
                  AppTheme.teal,
                ),
                _cDivider(),
                _confirmRow(
                  Icons.people_rounded,
                  'Occupancy',
                  '${_room?['beds']} person(s)',
                  AppTheme.textMuted,
                ),
                _cDivider(),
                _confirmRow(
                  Icons.event_rounded,
                  'Check-in',
                  '${_checkIn.day}/${_checkIn.month}/${_checkIn.year}',
                  AppTheme.textMuted,
                ),
                _cDivider(),
                _confirmRow(
                  Icons.date_range_rounded,
                  'Duration',
                  '$_duration month${_duration > 1 ? 's' : ''}',
                  AppTheme.textMuted,
                ),
                _cDivider(),
                _confirmRow(
                  Icons.person_rounded,
                  'Name',
                  _nameCtrl.text,
                  AppTheme.textMuted,
                ),
                _cDivider(),
                _confirmRow(
                  Icons.phone_rounded,
                  'Phone',
                  _phoneCtrl.text,
                  AppTheme.textMuted,
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),

          // Total price box
          Container(
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              gradient: const LinearGradient(
                colors: [AppTheme.tealDeep, AppTheme.teal],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
              borderRadius: BorderRadius.circular(18),
            ),
            child: Row(
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Total Amount',
                        style: GoogleFonts.dmSans(
                          color: Colors.white60,
                          fontSize: 12,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        'PKR ${_totalPrice.toString().replaceAllMapped(RegExp(r'(\d{1,3})(?=(\d{3})+(?!\d))'), (m) => '${m[1]},')}',
                        style: GoogleFonts.dmSans(
                          fontSize: 26,
                          fontWeight: FontWeight.w700,
                          color: Colors.white,
                        ),
                      ),
                      Text(
                        'for $_duration month${_duration > 1 ? 's' : ''}',
                        style: GoogleFonts.dmSans(
                          color: Colors.white60,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ),
                Container(
                  width: 56,
                  height: 56,
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.15),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.payments_rounded,
                    color: Colors.white,
                    size: 26,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),

          // Notice
          Container(
            padding: const EdgeInsets.all(13),
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.05),
              borderRadius: BorderRadius.circular(12),
              border: Border(left: BorderSide(color: AppTheme.teal, width: 3)),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(
                  Icons.info_outline_rounded,
                  color: AppTheme.teal,
                  size: 16,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'By confirming, you agree to the hostel\'s terms. The owner will review and approve your booking within 24 hours.',
                    style: GoogleFonts.dmSans(
                      fontSize: 12,
                      color: AppTheme.teal,
                      height: 1.4,
                    ),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),
        ],
      ),
    );
  }

  // ── Step 3: Success ────────────────────────────────────────────────────────
  Widget _buildSuccessStep() {
    return Center(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            // Animated checkmark
            TweenAnimationBuilder<double>(
              tween: Tween(begin: 0, end: 1),
              duration: const Duration(milliseconds: 800),
              curve: Curves.elasticOut,
              builder: (_, v, child) => Transform.scale(scale: v, child: child),
              child: Container(
                width: 100,
                height: 100,
                decoration: BoxDecoration(
                  color: AppTheme.green,
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: AppTheme.green.withOpacity(0.3),
                      blurRadius: 24,
                      spreadRadius: 4,
                    ),
                  ],
                ),
                child: const Icon(
                  Icons.check_rounded,
                  color: Colors.white,
                  size: 50,
                ),
              ),
            ),
            const SizedBox(height: 24),

            Text(
              'Booking Requested!',
              style: GoogleFonts.playfairDisplay(
                fontSize: 26,
                fontWeight: FontWeight.w700,
                color: AppTheme.textPrimary,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Your request has been sent to ${widget.hostelName}.\nThe owner will confirm within 24 hours.',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                fontSize: 14,
                color: AppTheme.textMuted,
                height: 1.5,
              ),
            ),
            const SizedBox(height: 20),

            // Booking ID
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
              decoration: BoxDecoration(
                color: AppTheme.card,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: AppTheme.border, width: 0.5),
              ),
              child: Column(
                children: [
                  Text(
                    'Booking Reference',
                    style: GoogleFonts.dmSans(
                      fontSize: 12,
                      color: AppTheme.textMuted,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    _bookingId ?? '',
                    style: GoogleFonts.dmSans(
                      fontSize: 22,
                      fontWeight: FontWeight.w700,
                      color: AppTheme.teal,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Summary chips
            Row(
              children: [
                Expanded(
                  child: _successChip(
                    Icons.meeting_room_rounded,
                    '${_room?['type']} Room',
                    AppTheme.teal,
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _successChip(
                    Icons.date_range_rounded,
                    '$_duration Month${_duration > 1 ? 's' : ''}',
                    const Color(0xFF185FA5),
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _successChip(
                    Icons.payments_rounded,
                    'PKR ${(_totalPrice / 1000).toStringAsFixed(0)}k',
                    AppTheme.green,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 28),

            SizedBox(
              width: double.infinity,
              height: 52,
              child: ElevatedButton(
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const MyBookingsScreen()),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.teal,
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(16),
                  ),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Icon(Icons.hotel_rounded, size: 18),
                    const SizedBox(width: 8),
                    Text(
                      'View My Bookings',
                      style: GoogleFonts.dmSans(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 12),
            TextButton(
              onPressed: () => Navigator.pushAndRemoveUntil(
                context,
                MaterialPageRoute(builder: (_) => const MyBookingsScreen()),
                (_) => false,
              ),
              child: Text(
                'Go to My Bookings',
                style: GoogleFonts.dmSans(
                  color: AppTheme.teal,
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ── Bottom bar ────────────────────────────────────────────────────────────
  Widget _buildBottomBar() {
    final labels = ['Continue to Details', 'Review Booking', 'Confirm Booking'];
    final icons = [
      Icons.arrow_forward_rounded,
      Icons.receipt_long_rounded,
      Icons.check_circle_rounded,
    ];

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
      child: SizedBox(
        width: double.infinity,
        height: 52,
        child: ElevatedButton(
          onPressed: _step == 2
              ? (_submitting ? null : _submitBooking)
              : _nextStep,
          style: ElevatedButton.styleFrom(
            backgroundColor: AppTheme.teal,
            foregroundColor: Colors.white,
            elevation: 0,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
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
                    Text(
                      labels[_step],
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
                      child: Icon(icons[_step], size: 14),
                    ),
                  ],
                ),
        ),
      ),
    );
  }

  // ── Helpers ───────────────────────────────────────────────────────────────
  Widget _roomState() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.teal.withOpacity(0.08),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: AppTheme.teal.withOpacity(0.30)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Icon(Icons.lock_open_rounded, color: AppTheme.teal),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              'Send your request to the owner first. After the owner shares your hostel credentials, you can select your floor, room and bed.',
              style: GoogleFonts.dmSans(
                fontSize: 12.5,
                height: 1.45,
                color: AppTheme.textPrimary,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionLabel(String text, IconData icon) => Padding(
    padding: const EdgeInsets.only(bottom: 0),
    child: Row(
      children: [
        Icon(icon, size: 14, color: AppTheme.teal),
        const SizedBox(width: 7),
        Text(
          text,
          style: GoogleFonts.dmSans(
            fontSize: 14,
            fontWeight: FontWeight.w700,
            color: AppTheme.textPrimary,
          ),
        ),
      ],
    ),
  );

  Widget _field(
    TextEditingController ctrl,
    String hint,
    IconData icon, {
    TextInputType? type,
    int maxLines = 1,
  }) {
    final error = _fieldErrors[ctrl];
    return TextField(
      controller: ctrl,
      keyboardType: type,
      maxLines: maxLines,
      onChanged: (_) {
        if (_fieldErrors.containsKey(ctrl)) {
          setState(() => _fieldErrors.remove(ctrl));
        }
      },
      style: GoogleFonts.dmSans(fontSize: 14, color: AppTheme.textPrimary),
      decoration: InputDecoration(
        filled: true,
        fillColor: AppTheme.card,
        hintText: hint,
        hintStyle: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13),
        prefixIcon: Icon(icon, size: 18, color: AppTheme.textMuted),
        errorText: error,
        errorStyle: GoogleFonts.dmSans(
          fontSize: 10,
          color: Colors.red.shade700,
        ),
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide(
            color: error == null ? AppTheme.border : Colors.red.shade400,
          ),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide(
            color: error == null ? AppTheme.border : Colors.red.shade400,
          ),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(14),
          borderSide: BorderSide(
            color: error == null ? AppTheme.teal : Colors.red.shade600,
            width: 1.4,
          ),
        ),
        contentPadding: const EdgeInsets.symmetric(
          horizontal: 16,
          vertical: 14,
        ),
      ),
    );
  }

  Widget _confirmRow(IconData icon, String label, String value, Color color) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Row(
        children: [
          Icon(icon, size: 16, color: color),
          const SizedBox(width: 10),
          Text(
            label,
            style: GoogleFonts.dmSans(fontSize: 13, color: AppTheme.textMuted),
          ),
          const Spacer(),
          Text(
            value,
            style: GoogleFonts.dmSans(
              fontSize: 13,
              fontWeight: FontWeight.w600,
              color: AppTheme.textPrimary,
            ),
            textAlign: TextAlign.right,
          ),
        ],
      ),
    );
  }

  Widget _cDivider() => Divider(
    height: 0.5,
    color: AppTheme.border,
    thickness: 0.5,
    indent: 16,
    endIndent: 16,
  );

  Widget _successChip(IconData icon, String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 12),
      decoration: BoxDecoration(
        color: color.withOpacity(0.06),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.2), width: 0.5),
      ),
      child: Column(
        children: [
          Icon(icon, size: 20, color: color),
          const SizedBox(height: 6),
          Text(
            text,
            style: GoogleFonts.dmSans(
              fontSize: 12,
              fontWeight: FontWeight.w700,
              color: color,
            ),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }
}
