import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme.dart';
import '../../../api.dart';
import '../../../widgets/responsive_shell.dart';

class AddStudentScreen extends StatefulWidget {
  final String hostelName;

  const AddStudentScreen({
    super.key,
    required this.hostelName,
  });

  @override
  State<AddStudentScreen> createState() => _AddStudentScreenState();
}

class _AddStudentScreenState extends State<AddStudentScreen> {
  final PageController _pageController = PageController();

  final _nameCtrl = TextEditingController();
  final _emailCtrl = TextEditingController();
  final _phoneCtrl = TextEditingController();
  final _cnicCtrl = TextEditingController();
  final _cityCtrl = TextEditingController();

  final _guardianNameCtrl = TextEditingController();
  final _guardianPhoneCtrl = TextEditingController();
  final _guardianEmailCtrl = TextEditingController();

  final _monthlyFeeCtrl = TextEditingController(text: '15000');

  int _currentStep = 0;
  bool _saving = false;

  String? _selectedRoom;
  String? _selectedBed;

  String _feeStatus = 'Paid';
  bool _createStudentLogin = true;
  bool _createParentLogin = true;

  final List<String> _rooms = [
    '101',
    '102',
    '201',
    '202',
    '301',
    '302',
  ];

  final Map<String, List<String>> _availableBeds = {
    '101': ['B1'],
    '102': ['B3'],
    '201': ['B2'],
    '202': [],
    '301': ['B1', 'B3'],
    '302': ['B1'],
  };

  static const int _totalSteps = 2;

  @override
  void dispose() {
    _pageController.dispose();

    _nameCtrl.dispose();
    _emailCtrl.dispose();
    _phoneCtrl.dispose();
    _cnicCtrl.dispose();
    _cityCtrl.dispose();

    _guardianNameCtrl.dispose();
    _guardianPhoneCtrl.dispose();
    _guardianEmailCtrl.dispose();

    _monthlyFeeCtrl.dispose();

    super.dispose();
  }

  // ===========================================================================
  // VALIDATION
  // ===========================================================================

  bool _validatePersonalInfo() {
    final name = _nameCtrl.text.trim();
    final email = _emailCtrl.text.trim();
    final phone = _phoneCtrl.text.trim();
    final cnic = _cnicCtrl.text.trim();
    final city = _cityCtrl.text.trim();

    if (name.isEmpty ||
        email.isEmpty ||
        phone.isEmpty ||
        cnic.isEmpty ||
        city.isEmpty) {
      _showError('Please fill all student information.');
      return false;
    }

    final emailRegex = RegExp(
      r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$',
    );

    if (!emailRegex.hasMatch(email)) {
      _showError('Please enter a valid student email address.');
      return false;
    }

    if (phone.length != 11 || !phone.startsWith('03')) {
      _showError(
        'Student phone number must contain 11 digits and start with 03.',
      );
      return false;
    }

    if (cnic.length != 13) {
      _showError('CNIC must contain exactly 13 digits.');
      return false;
    }

    return true;
  }

  bool _validateGuardian() {
    final name = _guardianNameCtrl.text.trim();
    final phone = _guardianPhoneCtrl.text.trim();

    if (name.isEmpty || phone.isEmpty) {
      _showError('Please provide guardian name and phone number.');
      return false;
    }

    if (phone.length != 11 || !phone.startsWith('03')) {
      _showError(
        'Guardian phone number must contain 11 digits and start with 03.',
      );
      return false;
    }

    final email = _guardianEmailCtrl.text.trim();

    if (email.isNotEmpty) {
      final emailRegex = RegExp(
        r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$',
      );

      if (!emailRegex.hasMatch(email)) {
        _showError('Please enter a valid guardian email address.');
        return false;
      }
    }

    return true;
  }

  bool _validateRoom() {
    if (_selectedRoom == null) {
      _showError('Please select a room.');
      return false;
    }

    if (_selectedBed == null) {
      _showError('Please select an available bed.');
      return false;
    }

    return true;
  }

  // ===========================================================================
  // NAVIGATION
  // ===========================================================================

  void _nextStep() {
    bool valid = true;

    if (_currentStep == 0) {
      valid = _validatePersonalInfo();
    } else if (_currentStep == 1) {
      valid = _validateGuardian();
    }

    if (!valid) return;

    if (_currentStep >= _totalSteps - 1) {
      _saveStudent();
      return;
    }

    setState(() {
      _currentStep++;
    });

    _pageController.animateToPage(
      _currentStep,
      duration: const Duration(milliseconds: 350),
      curve: Curves.easeOutCubic,
    );
  }

  void _previousStep() {
    if (_currentStep == 0) {
      Navigator.pop(context);
      return;
    }

    setState(() {
      _currentStep--;
    });

    _pageController.animateToPage(
      _currentStep,
      duration: const Duration(milliseconds: 320),
      curve: Curves.easeOutCubic,
    );
  }

  // ===========================================================================
  // SAVE STUDENT
  // ===========================================================================

  Future<void> _saveStudent() async {
  if (_saving) return;

  setState(() {
    _saving = true;
  });

  try {
   final response = await Api().addOwnerStudent(
  name: _nameCtrl.text.trim(),
  email: _emailCtrl.text.trim(),
  phone: _phoneCtrl.text.trim(),
  cnic: _cnicCtrl.text.trim(),
  city: _cityCtrl.text.trim(),
  parentName: _guardianNameCtrl.text.trim(),
  parentPhone: _guardianPhoneCtrl.text.trim(),
  parentEmail: _guardianEmailCtrl.text.trim(),
  parentRelation: 'Parent / Guardian',
  parentCity: _cityCtrl.text.trim(),
);

    if (!mounted) return;

    final studentRaw = response['student'];

    final student = studentRaw is Map
        ? Map<String, dynamic>.from(studentRaw)
        : <String, dynamic>{};

    // UI ke existing format ke liye mapping
    student['name'] ??= _nameCtrl.text.trim();
    student['email'] ??= _emailCtrl.text.trim();
    student['phone'] ??= _phoneCtrl.text.trim();

    student['room'] =
        student['room_number']?.toString().isNotEmpty == true
            ? student['room_number'].toString()
            : 'Unassigned';

    student['bed'] = student['bed_number']?.toString() ?? '-';
    student['status'] = 'Active';

    student['feeStatus'] = 'No invoice';

    student['parentLinked'] = false;

    student['guardian'] =
        _guardianNameCtrl.text.trim();

    student['guardianPhone'] =
        _guardianPhoneCtrl.text.trim();

    student['guardianEmail'] =
        _guardianEmailCtrl.text.trim();

    student['city'] =
        _cityCtrl.text.trim();

    student['cnic'] =
        _cnicCtrl.text.trim();

    student['joinDate'] =
        _formattedToday();

    student['temp_password'] =
        response['temp_password'];

    setState(() {
      _saving = false;
    });

    Navigator.pop(
      context,
      student,
    );
  } catch (e) {
    if (!mounted) return;

    setState(() {
      _saving = false;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          e.toString().replaceFirst(
                'Exception: ',
                '',
              ),
        ),
      ),
    );
  }
}
  String _formattedToday() {
    final now = DateTime.now();

    const months = [
      'Jan',
      'Feb',
      'Mar',
      'Apr',
      'May',
      'Jun',
      'Jul',
      'Aug',
      'Sep',
      'Oct',
      'Nov',
      'Dec',
    ];

    return '${now.day.toString().padLeft(2, '0')} '
        '${months[now.month - 1]} '
        '${now.year}';
  }

  // ===========================================================================
  // BUILD
  // ===========================================================================

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: AppTheme.bg,
        resizeToAvoidBottomInset: true,
        body: Column(
          children: [
            _buildHeader(),
            Expanded(
              child: PageView(
                controller: _pageController,
                physics: const NeverScrollableScrollPhysics(),
                children: [
                  _buildPersonalStep(),
                  _buildGuardianStep(),
                ],
              ),
            ),
            _buildBottomNavigation(),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // HEADER
  // ===========================================================================

  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Color(0xFF073F42),
            Color(0xFF0A696D),
            Color(0xFF0A797B),
          ],
        ),
        borderRadius: BorderRadius.vertical(
          bottom: Radius.circular(32),
        ),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(
            18,
            14,
            18,
            22,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  _circleButton(
                    icon: Icons.arrow_back_rounded,
                    onTap: _previousStep,
                  ),
                  const Spacer(),
                  Text(
                    '${_currentStep + 1} of $_totalSteps',
                    style: GoogleFonts.dmSans(
                      color: Colors.white.withOpacity(0.72),
                      fontSize: 11.5,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 18),

              _buildProgressIndicator(),

              const SizedBox(height: 18),

              Text(
                _stepTitle,
                style: GoogleFonts.playfairDisplay(
                  fontSize: 27,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),

              const SizedBox(height: 4),

              Text(
                _stepSubtitle,
                style: GoogleFonts.dmSans(
                  fontSize: 12,
                  color: Colors.white.withOpacity(0.65),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _circleButton({
    required IconData icon,
    required VoidCallback onTap,
  }) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: onTap,
        customBorder: const CircleBorder(),
        child: Container(
          width: 42,
          height: 42,
          decoration: BoxDecoration(
            color: Colors.white.withOpacity(0.13),
            shape: BoxShape.circle,
            border: Border.all(
              color: Colors.white.withOpacity(0.18),
            ),
          ),
          child: Icon(
            icon,
            color: Colors.white,
            size: 20,
          ),
        ),
      ),
    );
  }

  Widget _buildProgressIndicator() {
    return Row(
      children: List.generate(
        _totalSteps,
        (index) {
          final active = index <= _currentStep;

          return Expanded(
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 250),
              height: index == _currentStep ? 5 : 4,
              margin: EdgeInsets.only(
                right: index < _totalSteps - 1 ? 7 : 0,
              ),
              decoration: BoxDecoration(
                color: active
                    ? Colors.white
                    : Colors.white.withOpacity(0.22),
                borderRadius: BorderRadius.circular(8),
              ),
            ),
          );
        },
      ),
    );
  }

  String get _stepTitle {
    switch (_currentStep) {
      case 0:
        return 'Student Information';
      case 1:
        return 'Guardian Details';
      case 2:
        return 'Room Assignment';
      default:
        return 'Review & Create';
    }
  }

  String get _stepSubtitle {
    switch (_currentStep) {
      case 0:
        return 'Enter the student\'s basic information';
      case 1:
        return 'Add parent or guardian contact details';
      case 2:
        return 'Assign an available room and bed';
      default:
        return 'Review everything before adding the student';
    }
  }

  // ===========================================================================
  // STEP 1 - PERSONAL INFO
  // ===========================================================================

  Widget _buildPersonalStep() {
    return _stepScrollView(
      children: [
        _introCard(
          icon: Icons.person_add_alt_1_rounded,
          title: 'Student profile',
          subtitle:
              'This information will appear across hostel management records.',
        ),

        const SizedBox(height: 20),

        _sectionHeading(
          'Personal information',
        ),

        const SizedBox(height: 12),

        _field(
          controller: _nameCtrl,
          label: 'Full name',
          hint: 'e.g. Sara Ahmed',
          icon: Icons.person_outline_rounded,
          capitalization: TextCapitalization.words,
        ),

        const SizedBox(height: 12),

        _field(
          controller: _emailCtrl,
          label: 'Email address',
          hint: 'student@example.com',
          icon: Icons.email_outlined,
          keyboardType: TextInputType.emailAddress,
        ),

        const SizedBox(height: 12),

        _field(
          controller: _phoneCtrl,
          label: 'Phone number',
          hint: '03XXXXXXXXX',
          icon: Icons.phone_outlined,
          keyboardType: TextInputType.phone,
          formatters: [
            FilteringTextInputFormatter.digitsOnly,
            LengthLimitingTextInputFormatter(11),
          ],
        ),

        const SizedBox(height: 12),

        _field(
          controller: _cnicCtrl,
          label: 'CNIC',
          hint: '13 digit CNIC',
          icon: Icons.badge_outlined,
          keyboardType: TextInputType.number,
          formatters: [
            FilteringTextInputFormatter.digitsOnly,
            LengthLimitingTextInputFormatter(13),
          ],
        ),

        const SizedBox(height: 12),

        _field(
          controller: _cityCtrl,
          label: 'City',
          hint: 'e.g. Lahore',
          icon: Icons.location_city_outlined,
          capitalization: TextCapitalization.words,
        ),
      ],
    );
  }

  // ===========================================================================
  // STEP 2 - GUARDIAN
  // ===========================================================================

  Widget _buildGuardianStep() {
    return _stepScrollView(
      children: [
        _introCard(
          icon: Icons.family_restroom_rounded,
          title: 'Guardian information',
          subtitle: 'Guardian details are useful for emergencies.',
        ),

        const SizedBox(height: 20),

        _sectionHeading(
          'Primary guardian',
        ),

        const SizedBox(height: 12),

        _field(
          controller: _guardianNameCtrl,
          label: 'Guardian name',
          hint: 'Parent or guardian full name',
          icon: Icons.person_outline_rounded,
          capitalization: TextCapitalization.words,
        ),

        const SizedBox(height: 12),

        _field(
          controller: _guardianPhoneCtrl,
          label: 'Guardian phone',
          hint: '03XXXXXXXXX',
          icon: Icons.phone_outlined,
          keyboardType: TextInputType.phone,
          formatters: [
            FilteringTextInputFormatter.digitsOnly,
            LengthLimitingTextInputFormatter(11),
          ],
        ),

        const SizedBox(height: 12),

        _field(
          controller: _guardianEmailCtrl,
          label: 'Guardian email',
          hint: 'Optional',
          icon: Icons.email_outlined,
          keyboardType: TextInputType.emailAddress,
        ),

      ],
    );
  }

  // ===========================================================================
  // STEP 3 - ROOM
  // ===========================================================================

  Widget _buildRoomStep() {
    final beds = _selectedRoom == null
        ? <String>[]
        : (_availableBeds[_selectedRoom] ?? <String>[]);

    return _stepScrollView(
      children: [
        _introCard(
          icon: Icons.bed_rounded,
          title: 'Accommodation',
          subtitle:
              'Choose the room and bed that will be assigned to this student.',
        ),

        const SizedBox(height: 20),

        _sectionHeading(
          'Select room',
        ),

        const SizedBox(height: 12),

        _dropdown(
          value: _selectedRoom,
          hint: 'Choose room',
          icon: Icons.meeting_room_outlined,
          items: _rooms,
          onChanged: (value) {
            setState(() {
              _selectedRoom = value;
              _selectedBed = null;
            });
          },
        ),

        const SizedBox(height: 16),

        if (_selectedRoom != null) ...[
          _roomAvailabilityCard(
            room: _selectedRoom!,
            beds: beds,
          ),

          const SizedBox(height: 20),

          _sectionHeading(
            'Select bed',
          ),

          const SizedBox(height: 12),

          if (beds.isEmpty)
            _noBedsAvailableCard()
          else
            _buildBedSelector(
              beds,
            ),
        ],

        const SizedBox(height: 22),

        _sectionHeading(
          'Fee information',
        ),

        const SizedBox(height: 12),

        _field(
          controller: _monthlyFeeCtrl,
          label: 'Monthly hostel fee',
          hint: '15000',
          icon: Icons.payments_outlined,
          keyboardType: TextInputType.number,
          formatters: [
            FilteringTextInputFormatter.digitsOnly,
          ],
        ),

        const SizedBox(height: 12),

        _dropdown(
          value: _feeStatus,
          hint: 'Fee status',
          icon: Icons.receipt_long_outlined,
          items: const [
            'Paid',
            'Due',
          ],
          onChanged: (value) {
            if (value == null) return;

            setState(() {
              _feeStatus = value;
            });
          },
        ),
      ],
    );
  }

  Widget _roomAvailabilityCard({
    required String room,
    required List<String> beds,
  }) {
    final hasBeds = beds.isNotEmpty;

    final color = hasBeds
        ? const Color(0xFF3D7B53)
        : const Color(0xFFC07421);

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: AppTheme.border,
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 46,
            height: 46,
            decoration: BoxDecoration(
              color: color.withOpacity(0.09),
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(
              Icons.bed_rounded,
              size: 22,
              color: color,
            ),
          ),

          const SizedBox(width: 12),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Room $room',
                  style: GoogleFonts.dmSans(
                    fontSize: 14,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  hasBeds
                      ? '${beds.length} bed${beds.length == 1 ? '' : 's'} available'
                      : 'Currently full',
                  style: GoogleFonts.dmSans(
                    fontSize: 10.5,
                    color: AppTheme.textMuted,
                  ),
                ),
              ],
            ),
          ),

          Container(
            padding: const EdgeInsets.symmetric(
              horizontal: 9,
              vertical: 5,
            ),
            decoration: BoxDecoration(
              color: color.withOpacity(0.09),
              borderRadius: BorderRadius.circular(20),
            ),
            child: Text(
              hasBeds ? 'Available' : 'Full',
              style: GoogleFonts.dmSans(
                fontSize: 9,
                fontWeight: FontWeight.w700,
                color: color,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBedSelector(
    List<String> beds,
  ) {
    return Wrap(
      spacing: 10,
      runSpacing: 10,
      children: beds.map((bed) {
        final selected = _selectedBed == bed;

        return GestureDetector(
          onTap: () {
            setState(() {
              _selectedBed = bed;
            });
          },
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 180),
            width: 78,
            height: 70,
            decoration: BoxDecoration(
              color: selected
                  ? AppTheme.teal
                  : AppTheme.card,
              borderRadius: BorderRadius.circular(17),
              border: Border.all(
                color: selected
                    ? AppTheme.teal
                    : AppTheme.border,
              ),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.bed_rounded,
                  size: 22,
                  color: selected
                      ? Colors.white
                      : AppTheme.teal,
                ),
                const SizedBox(height: 5),
                Text(
                  bed,
                  style: GoogleFonts.dmSans(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: selected
                        ? Colors.white
                        : AppTheme.textPrimary,
                  ),
                ),
              ],
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _noBedsAvailableCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: const Color(0xFFFFF4E6),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: const Color(0xFFC07421).withOpacity(0.20),
        ),
      ),
      child: Row(
        children: [
          const Icon(
            Icons.warning_amber_rounded,
            color: Color(0xFFC07421),
            size: 20,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              'This room has no available beds. Please select another room.',
              style: GoogleFonts.dmSans(
                color: const Color(0xFFC07421),
                fontSize: 11,
                height: 1.4,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // STEP 4 - REVIEW
  // ===========================================================================

  Widget _buildReviewStep() {
    return _stepScrollView(
      children: [
        _introCard(
          icon: Icons.fact_check_rounded,
          title: 'Almost done',
          subtitle:
              'Review the student information before creating the record.',
        ),

        const SizedBox(height: 20),

        _reviewSection(
          title: 'Student',
          icon: Icons.person_outline_rounded,
          children: [
            _reviewRow(
              'Name',
              _valueOrDash(_nameCtrl.text),
            ),
            _reviewRow(
              'Email',
              _valueOrDash(_emailCtrl.text),
            ),
            _reviewRow(
              'Phone',
              _valueOrDash(_phoneCtrl.text),
            ),
            _reviewRow(
              'CNIC',
              _valueOrDash(_cnicCtrl.text),
            ),
            _reviewRow(
              'City',
              _valueOrDash(_cityCtrl.text),
            ),
          ],
        ),

        const SizedBox(height: 14),

        _reviewSection(
          title: 'Guardian',
          icon: Icons.family_restroom_rounded,
          children: [
            _reviewRow(
              'Name',
              _valueOrDash(_guardianNameCtrl.text),
            ),
            _reviewRow(
              'Phone',
              _valueOrDash(_guardianPhoneCtrl.text),
            ),
            _reviewRow(
              'Email',
              _valueOrDash(_guardianEmailCtrl.text),
            ),
          ],
        ),

        const SizedBox(height: 14),

        _reviewSection(
          title: 'Accommodation',
          icon: Icons.bed_rounded,
          children: [
            _reviewRow(
              'Hostel',
              widget.hostelName,
            ),
            _reviewRow(
              'Room',
              _selectedRoom ?? 'Not selected',
            ),
            _reviewRow(
              'Bed',
              _selectedBed ?? 'Not selected',
            ),
            _reviewRow(
              'Monthly fee',
              'PKR ${_monthlyFeeCtrl.text.trim().isEmpty ? '0' : _monthlyFeeCtrl.text.trim()}',
            ),
            _reviewRow(
              'Fee status',
              _feeStatus,
            ),
          ],
        ),

        const SizedBox(height: 14),

        _switchCard(
          icon: Icons.password_rounded,
          title: 'Generate student credentials',
          subtitle:
              'Prepare a StayBuddy login account for this student.',
          value: _createStudentLogin,
          onChanged: (value) {
            setState(() {
              _createStudentLogin = value;
            });
          },
        ),

        const SizedBox(height: 12),

        _switchCard(
          icon: Icons.family_restroom_rounded,
          title: 'Prepare parent invitation',
          subtitle:
              'Create parent access after the student record is created.',
          value: _createParentLogin,
          onChanged: (value) {
            setState(() {
              _createParentLogin = value;
            });
          },
        ),

        const SizedBox(height: 18),

        Container(
          padding: const EdgeInsets.all(15),
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.06),
            borderRadius: BorderRadius.circular(17),
            border: Border.all(
              color: AppTheme.teal.withOpacity(0.15),
            ),
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(
                Icons.info_outline_rounded,
                color: AppTheme.teal,
                size: 19,
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'For now this screen saves the student to the local UI only. '
                  'Later we will connect the same form to your backend API.',
                  style: GoogleFonts.dmSans(
                    fontSize: 10.5,
                    height: 1.45,
                    color: AppTheme.teal,
                  ),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _reviewSection({
    required String title,
    required IconData icon,
    required List<Widget> children,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(17),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.border,
        ),
      ),
      child: Column(
        children: [
          Row(
            children: [
              Container(
                width: 39,
                height: 39,
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.07),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(
                  icon,
                  color: AppTheme.teal,
                  size: 19,
                ),
              ),
              const SizedBox(width: 11),
              Expanded(
                child: Text(
                  title,
                  style: GoogleFonts.dmSans(
                    fontSize: 13.5,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          ...children,
        ],
      ),
    );
  }

  Widget _reviewRow(
    String label,
    String value,
  ) {
    return Padding(
      padding: const EdgeInsets.symmetric(
        vertical: 6,
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 95,
            child: Text(
              label,
              style: GoogleFonts.dmSans(
                fontSize: 10.5,
                color: AppTheme.textMuted,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              value,
              textAlign: TextAlign.right,
              style: GoogleFonts.dmSans(
                fontSize: 11,
                fontWeight: FontWeight.w600,
                color: AppTheme.textPrimary,
              ),
            ),
          ),
        ],
      ),
    );
  }

  String _valueOrDash(
    String value,
  ) {
    final trimmed = value.trim();

    return trimmed.isEmpty ? '-' : trimmed;
  }

  // ===========================================================================
  // COMMON COMPONENTS
  // ===========================================================================

  Widget _stepScrollView({
    required List<Widget> children,
  }) {
    return SingleChildScrollView(
      keyboardDismissBehavior:
          ScrollViewKeyboardDismissBehavior.onDrag,
      padding: const EdgeInsets.fromLTRB(
        18,
        20,
        18,
        30,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: children,
      ),
    );
  }

  Widget _introCard({
    required IconData icon,
    required String title,
    required String subtitle,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(17),
      decoration: BoxDecoration(
        color: AppTheme.teal.withOpacity(0.06),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: AppTheme.teal.withOpacity(0.13),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 48,
            height: 48,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.10),
              borderRadius: BorderRadius.circular(15),
            ),
            child: Icon(
              icon,
              size: 23,
              color: AppTheme.teal,
            ),
          ),

          const SizedBox(width: 13),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.dmSans(
                    fontSize: 13.5,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.textPrimary,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  subtitle,
                  style: GoogleFonts.dmSans(
                    fontSize: 10.5,
                    height: 1.4,
                    color: AppTheme.textMuted,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionHeading(
    String text,
  ) {
    return Text(
      text,
      style: GoogleFonts.playfairDisplay(
        fontSize: 18,
        fontWeight: FontWeight.w700,
        color: AppTheme.textPrimary,
      ),
    );
  }

  Widget _field({
    required TextEditingController controller,
    required String label,
    required String hint,
    required IconData icon,
    TextInputType? keyboardType,
    List<TextInputFormatter>? formatters,
    TextCapitalization capitalization = TextCapitalization.none,
  }) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: GoogleFonts.dmSans(
            fontSize: 10.5,
            fontWeight: FontWeight.w600,
            color: AppTheme.textMuted,
          ),
        ),

        const SizedBox(height: 6),

        Container(
          decoration: BoxDecoration(
            color: AppTheme.card,
            borderRadius: BorderRadius.circular(16),
            border: Border.all(
              color: AppTheme.border,
              width: 0.7,
            ),
          ),
          child: TextField(
            controller: controller,
            keyboardType: keyboardType,
            inputFormatters: formatters,
            textCapitalization: capitalization,
            style: GoogleFonts.dmSans(
              fontSize: 13.5,
              color: AppTheme.textPrimary,
            ),
            decoration: InputDecoration(
              hintText: hint,
              hintStyle: GoogleFonts.dmSans(
                color: AppTheme.textMuted,
                fontSize: 12.5,
              ),
              prefixIcon: Icon(
                icon,
                size: 20,
                color: AppTheme.textMuted,
              ),
              border: InputBorder.none,
              contentPadding: const EdgeInsets.symmetric(
                horizontal: 15,
                vertical: 16,
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _dropdown({
    required String? value,
    required String hint,
    required IconData icon,
    required List<String> items,
    required ValueChanged<String?> onChanged,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 13,
      ),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: AppTheme.border,
        ),
      ),
      child: Row(
        children: [
          Icon(
            icon,
            size: 20,
            color: AppTheme.textMuted,
          ),

          const SizedBox(width: 12),

          Expanded(
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                value: value,
                hint: Text(
                  hint,
                  style: GoogleFonts.dmSans(
                    color: AppTheme.textMuted,
                    fontSize: 12.5,
                  ),
                ),
                isExpanded: true,
                borderRadius: BorderRadius.circular(18),
                items: items.map((item) {
                  return DropdownMenuItem(
                    value: item,
                     child: Text(
                       item,
                       maxLines: 1,
                       overflow: TextOverflow.ellipsis,
                       style: GoogleFonts.dmSans(
                        fontSize: 13,
                        color: AppTheme.textPrimary,
                      ),
                    ),
                  );
                }).toList(),
                onChanged: onChanged,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _switchCard({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Container(
      padding: const EdgeInsets.all(15),
      decoration: BoxDecoration(
        color: AppTheme.card,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: AppTheme.border,
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 43,
            height: 43,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.07),
              borderRadius: BorderRadius.circular(13),
            ),
            child: Icon(
              icon,
              color: AppTheme.teal,
              size: 20,
            ),
          ),

          const SizedBox(width: 12),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: GoogleFonts.dmSans(
                    fontSize: 12.5,
                    fontWeight: FontWeight.w600,
                    color: AppTheme.textPrimary,
                  ),
                ),

                const SizedBox(height: 2),

                Text(
                  subtitle,
                  style: GoogleFonts.dmSans(
                    fontSize: 9.5,
                    height: 1.4,
                    color: AppTheme.textMuted,
                  ),
                ),
              ],
            ),
          ),

          Switch.adaptive(
            value: value,
            activeColor: AppTheme.teal,
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // BOTTOM ACTION
  // ===========================================================================

  Widget _buildBottomNavigation() {
    return SafeArea(
      top: false,
      child: Container(
        padding: const EdgeInsets.fromLTRB(
          18,
          12,
          18,
          14,
        ),
        decoration: BoxDecoration(
          color: AppTheme.bg,
          border: Border(
            top: BorderSide(
              color: AppTheme.border,
              width: 0.7,
            ),
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.035),
              blurRadius: 20,
              offset: const Offset(0, -5),
            ),
          ],
        ),
        child: Row(
          children: [
            if (_currentStep > 0) ...[
              SizedBox(
                width: 52,
                height: 52,
                child: OutlinedButton(
                  onPressed: _saving
                      ? null
                      : _previousStep,
                  style: OutlinedButton.styleFrom(
                    padding: EdgeInsets.zero,
                    foregroundColor: AppTheme.teal,
                    side: BorderSide(
                      color: AppTheme.teal.withOpacity(0.40),
                    ),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                  child: const Icon(
                    Icons.arrow_back_rounded,
                  ),
                ),
              ),
              const SizedBox(width: 10),
            ],

            Expanded(
              child: SizedBox(
                height: 52,
                child: ElevatedButton(
                  onPressed: _saving
                      ? null
                      : _nextStep,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: AppTheme.teal,
                    disabledBackgroundColor:
                        AppTheme.teal.withOpacity(0.60),
                    foregroundColor: Colors.white,
                    elevation: 0,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(16),
                    ),
                  ),
                  child: _saving
                      ? const SizedBox(
                          width: 21,
                          height: 21,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : Row(
                          mainAxisAlignment:
                              MainAxisAlignment.center,
                          children: [
                            Text(
                              _currentStep ==
                                      _totalSteps - 1
                                  ? 'Create Student'
                                  : 'Continue',
                              style: GoogleFonts.dmSans(
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                              ),
                            ),
                            const SizedBox(width: 8),
                            Icon(
                              _currentStep ==
                                      _totalSteps - 1
                                  ? Icons.check_rounded
                                  : Icons.arrow_forward_rounded,
                              size: 18,
                            ),
                          ],
                        ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // ERROR
  // ===========================================================================

  void _showError(
    String message,
  ) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: Colors.red.shade700,
          margin: const EdgeInsets.all(16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          content: Row(
            children: [
              const Icon(
                Icons.error_outline_rounded,
                color: Colors.white,
                size: 18,
              ),
              const SizedBox(width: 9),
              Expanded(
                child: Text(
                  message,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 12,
                  ),
                ),
              ),
            ],
          ),
        ),
      );
  }
}
