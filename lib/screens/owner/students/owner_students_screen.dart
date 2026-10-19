import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme.dart';
import '../../../api.dart';
import '../../../widgets/owner_feature_frame.dart';
import '../../../widgets/responsive_shell.dart';
import 'add_student_screen.dart';
import 'student_details_screen.dart';

class OwnerStudentsScreen extends StatefulWidget {
  final String hostelName;
  final VoidCallback? onBackToDashboard;

  const OwnerStudentsScreen({
    super.key,
    required this.hostelName,
    this.onBackToDashboard,
  });

  @override
  State<OwnerStudentsScreen> createState() => _OwnerStudentsScreenState();
}

class _OwnerStudentsScreenState extends State<OwnerStudentsScreen> {
  final TextEditingController _searchCtrl = TextEditingController();

  String _selectedFilter = 'All';
  String _selectedRoomFilter = 'All Rooms';

  final List<String> _filters = ['All', 'Active', 'Fee Due', 'Unassigned'];

  final List<String> _roomFilters = [
    'All Rooms',
    '101',
    '102',
    '201',
    '202',
    '301',
  ];

  List<Map<String, dynamic>> _students = [];
  bool _isLoadingStudents = true;
  String? _studentsError;
  @override
  void initState() {
    super.initState();
    _loadStudents();
  }

  Future<void> _loadStudents() async {
    try {
      final result = await Api().getOwnerStudents();

      if (!mounted) return;

      setState(() {
        _students = result.whereType<Map>().map((item) {
          final data = Map<String, dynamic>.from(item);

          return {
            'id': data['id'],
            'bookingId': data['booking_id'],
            'bookingStatus': data['booking_status']?.toString() ?? '',
            'name': data['name']?.toString() ?? 'Student',
            'email': data['email']?.toString() ?? '',
            'phone': data['phone']?.toString() ?? '',
            'cnic': data['student_cnic']?.toString() ?? '',
            'university': data['university']?.toString() ?? '',
            'department': data['department']?.toString() ?? '',
            'semester': data['semester']?.toString() ?? '',
            'guardian': data['parent_name']?.toString() ?? '',
            'parentPhone': data['parent_phone']?.toString() ?? '',
            'parentRelation': data['parent_relation']?.toString() ?? '',
            'parentEmail': data['parent_email']?.toString() ?? '',
            'requirements': data['special_requirements']?.toString() ?? '',

            'room': data['room_number']?.toString().trim().isNotEmpty == true
                ? data['room_number'].toString()
                : 'Unassigned',

            'bed': data['bed_number']?.toString().trim().isNotEmpty == true
                ? data['bed_number'].toString()
                : '-',
            'status':
                data['booking_status']?.toString().toLowerCase() == 'confirmed'
                ? 'Active'
                : 'Inactive',
            'feeStatus': switch (data['fee_status']?.toString()) {
              'paid' => 'Paid',
              'pending' || 'overdue' => 'Due',
              _ => 'No invoice',
            },
            'parentLinked': false,
            'city': data['parent_city']?.toString() ?? '',
            'joinDate': '',
          };
        }).toList();

        _isLoadingStudents = false;
        _studentsError = null;
      });
    } catch (e) {
      if (!mounted) return;

      setState(() {
        _isLoadingStudents = false;
        _studentsError = e.toString();
      });
    }
  }

  @override
  void dispose() {
    _searchCtrl.dispose();
    super.dispose();
  }

  List<Map<String, dynamic>> get _filteredStudents {
    final query = _searchCtrl.text.trim().toLowerCase();

    return _students.where((student) {
      final name = student['name'].toString().toLowerCase();
      final email = student['email'].toString().toLowerCase();
      final room = student['room'].toString();

      final matchesSearch =
          query.isEmpty ||
          name.contains(query) ||
          email.contains(query) ||
          room.toLowerCase().contains(query);

      bool matchesFilter = true;

      if (_selectedFilter == 'Active') {
        matchesFilter = student['status'] == 'Active';
      } else if (_selectedFilter == 'Fee Due') {
        matchesFilter = student['feeStatus'] == 'Due';
      } else if (_selectedFilter == 'Unassigned') {
        matchesFilter = student['room'] == 'Unassigned';
      }

      final matchesRoom =
          _selectedRoomFilter == 'All Rooms' ||
          student['room'] == _selectedRoomFilter;

      return matchesSearch && matchesFilter && matchesRoom;
    }).toList();
  }

  int get _totalStudents => _students.length;

  int get _activeStudents =>
      _students.where((s) => s['status'] == 'Active').length;

  int get _feeDueStudents =>
      _students.where((s) => s['feeStatus'] == 'Due').length;

  int get _unassignedStudents =>
      _students.where((s) => s['room'] == 'Unassigned').length;

  @override
  Widget build(BuildContext context) {
    if (_isLoadingStudents) {
      return const ResponsiveShell(
        child: Scaffold(
          backgroundColor: Colors.transparent,
          body: OwnerFeatureFrame(
            child: Center(
              child: CircularProgressIndicator(color: AppTheme.tealMint),
            ),
          ),
        ),
      );
    }

    if (_studentsError != null) {
      return ResponsiveShell(
        child: Scaffold(
          backgroundColor: Colors.transparent,
          body: OwnerFeatureFrame(
            child: Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(
                      Icons.error_outline_rounded,
                      size: 42,
                      color: AppTheme.tealMint,
                    ),
                    const SizedBox(height: 12),
                    const Text(
                      'Could not load students',
                      style: TextStyle(color: Colors.white),
                    ),
                    const SizedBox(height: 12),
                    ElevatedButton(
                      onPressed: _loadStudents,
                      child: const Text('Retry'),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      );
    }
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        floatingActionButton: _buildFab(),
        body: OwnerFeatureFrame(
          child: CustomScrollView(
            physics: const BouncingScrollPhysics(),
            slivers: [
              SliverToBoxAdapter(child: _buildHeader()),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 18, 18, 0),
                sliver: SliverToBoxAdapter(child: _buildSummary()),
              ),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 22, 18, 0),
                sliver: SliverToBoxAdapter(child: _buildSearchBar()),
              ),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 14, 18, 0),
                sliver: SliverToBoxAdapter(child: _buildFilters()),
              ),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 14, 18, 0),
                sliver: SliverToBoxAdapter(child: _buildRoomFilter()),
              ),
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(18, 24, 18, 10),
                sliver: SliverToBoxAdapter(child: _buildSectionHeader()),
              ),
              _buildStudentsSliver(),
              const SliverToBoxAdapter(child: SizedBox(height: 110)),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF073F42), Color(0xFF0A696D), Color(0xFF0A797B)],
        ),
        borderRadius: BorderRadius.vertical(bottom: Radius.circular(32)),
      ),
      child: SafeArea(
        bottom: false,
        child: Padding(
          padding: const EdgeInsets.fromLTRB(18, 14, 18, 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  _circleButton(
                    icon: Icons.arrow_back_rounded,
                    onTap: _goBackToDashboard,
                  ),
                  const Spacer(),
                  _circleButton(
                    icon: Icons.more_horiz_rounded,
                    onTap: _showMoreMenu,
                  ),
                ],
              ),
              const SizedBox(height: 22),
              Text(
                'Students',
                style: GoogleFonts.playfairDisplay(
                  fontSize: 29,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 5),
              Text(
                widget.hostelName,
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
                style: GoogleFonts.dmSans(
                  fontSize: 12.5,
                  color: Colors.white.withOpacity(0.65),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _circleButton({required IconData icon, required VoidCallback onTap}) {
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
            border: Border.all(color: Colors.white.withOpacity(0.18)),
          ),
          child: Icon(icon, color: Colors.white, size: 20),
        ),
      ),
    );
  }

  void _goBackToDashboard() {
    if (widget.onBackToDashboard != null) {
      widget.onBackToDashboard!();
      return;
    }
    Navigator.of(context).maybePop();
  }

  Widget _buildSummary() {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border, width: 0.6),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.035),
            blurRadius: 16,
            offset: const Offset(0, 7),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Student Overview',
            style: GoogleFonts.dmSans(
              fontSize: 14,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                child: _summaryItem(
                  value: '$_totalStudents',
                  label: 'Total',
                  icon: Icons.groups_rounded,
                  color: AppTheme.teal,
                ),
              ),
              _summaryDivider(),
              Expanded(
                child: _summaryItem(
                  value: '$_activeStudents',
                  label: 'Active',
                  icon: Icons.verified_user_rounded,
                  color: const Color(0xFF3D7B53),
                ),
              ),
              _summaryDivider(),
              Expanded(
                child: _summaryItem(
                  value: '$_feeDueStudents',
                  label: 'Fee Due',
                  icon: Icons.receipt_long_rounded,
                  color: const Color(0xFFC07421),
                ),
              ),
              _summaryDivider(),
              Expanded(
                child: _summaryItem(
                  value: '$_unassignedStudents',
                  label: 'Unassigned',
                  icon: Icons.bed_outlined,
                  color: const Color(0xFF7D55B4),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _summaryItem({
    required String value,
    required String label,
    required IconData icon,
    required Color color,
  }) {
    return Column(
      children: [
        Icon(icon, size: 18, color: color),
        const SizedBox(height: 6),
        Text(
          value,
          style: GoogleFonts.dmSans(
            fontSize: 17,
            fontWeight: FontWeight.w700,
            color: OwnerFeaturePalette.text,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(
            fontSize: 9.5,
            color: OwnerFeaturePalette.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _summaryDivider() {
    return Container(width: 1, height: 42, color: OwnerFeaturePalette.border);
  }

  Widget _buildSearchBar() {
    return Container(
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: OwnerFeaturePalette.border, width: 0.7),
      ),
      child: TextField(
        controller: _searchCtrl,
        onChanged: (_) => setState(() {}),
        style: GoogleFonts.dmSans(
          fontSize: 13.5,
          color: OwnerFeaturePalette.text,
        ),
        decoration: InputDecoration(
          hintText: 'Search student, email or room...',
          hintStyle: GoogleFonts.dmSans(
            color: OwnerFeaturePalette.textMuted,
            fontSize: 12.5,
          ),
          prefixIcon: Icon(
            Icons.search_rounded,
            color: OwnerFeaturePalette.textMuted,
            size: 20,
          ),
          suffixIcon: _searchCtrl.text.isEmpty
              ? null
              : IconButton(
                  icon: const Icon(Icons.close_rounded, size: 18),
                  onPressed: () {
                    _searchCtrl.clear();
                    setState(() {});
                  },
                ),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(vertical: 15),
        ),
      ),
    );
  }

  Widget _buildFilters() {
    return SizedBox(
      height: 38,
      child: ListView.separated(
        scrollDirection: Axis.horizontal,
        itemCount: _filters.length,
        separatorBuilder: (_, __) => const SizedBox(width: 8),
        itemBuilder: (context, index) {
          final filter = _filters[index];
          final selected = filter == _selectedFilter;

          return GestureDetector(
            onTap: () {
              setState(() {
                _selectedFilter = filter;
              });
            },
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 180),
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
              decoration: BoxDecoration(
                color: selected ? AppTheme.teal : OwnerFeaturePalette.surface,
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: selected ? AppTheme.teal : OwnerFeaturePalette.border,
                ),
              ),
              child: Text(
                filter,
                style: GoogleFonts.dmSans(
                  fontSize: 11,
                  fontWeight: FontWeight.w600,
                  color: selected
                      ? Colors.white
                      : OwnerFeaturePalette.textMuted,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildRoomFilter() {
    return Row(
      children: [
        Icon(
          Icons.meeting_room_outlined,
          size: 18,
          color: OwnerFeaturePalette.textMuted,
        ),
        const SizedBox(width: 8),
        Text(
          'Room:',
          style: GoogleFonts.dmSans(
            fontSize: 12,
            color: OwnerFeaturePalette.textMuted,
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: DropdownButtonHideUnderline(
            child: DropdownButton<String>(
              value: _selectedRoomFilter,
              isExpanded: true,
              borderRadius: BorderRadius.circular(16),
              items: _roomFilters.map((room) {
                return DropdownMenuItem(
                  value: room,
                  child: Text(
                    room,
                    style: GoogleFonts.dmSans(
                      fontSize: 12.5,
                      color: OwnerFeaturePalette.text,
                    ),
                  ),
                );
              }).toList(),
              onChanged: (value) {
                if (value == null) return;

                setState(() {
                  _selectedRoomFilter = value;
                });
              },
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildSectionHeader() {
    return Row(
      children: [
        Expanded(
          child: Text(
            'All Students',
            style: GoogleFonts.playfairDisplay(
              fontSize: 19,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),
        ),
        Text(
          '${_filteredStudents.length} found',
          style: GoogleFonts.dmSans(fontSize: 10.5, color: Colors.white70),
        ),
      ],
    );
  }

  Widget _buildStudentsSliver() {
    final students = _filteredStudents;

    if (students.isEmpty) {
      return SliverPadding(
        padding: const EdgeInsets.fromLTRB(18, 16, 18, 0),
        sliver: SliverToBoxAdapter(child: _buildEmptyState()),
      );
    }

    return SliverPadding(
      padding: const EdgeInsets.symmetric(horizontal: 18),
      sliver: SliverList(
        delegate: SliverChildBuilderDelegate((context, index) {
          final student = students[index];

          return Padding(
            padding: const EdgeInsets.only(bottom: 12),
            child: _buildStudentCard(student),
          );
        }, childCount: students.length),
      ),
    );
  }

  Widget _buildStudentCard(Map<String, dynamic> student) {
    final feeDue = student['feeStatus'] == 'Due';
    final active = student['status'] == 'Active';

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () => _openStudentDetails(student),
        borderRadius: BorderRadius.circular(20),
        child: Container(
          padding: const EdgeInsets.all(15),
          decoration: BoxDecoration(
            color: OwnerFeaturePalette.surface,
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: OwnerFeaturePalette.border, width: 0.6),
          ),
          child: Column(
            children: [
              Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _studentAvatar(student['name']),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Flexible(
                              child: Text(
                                student['name'],
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: GoogleFonts.dmSans(
                                  fontSize: 14,
                                  fontWeight: FontWeight.w700,
                                  color: OwnerFeaturePalette.text,
                                ),
                              ),
                            ),
                            const SizedBox(width: 8),
                            _statusChip(
                              active ? 'Active' : 'Inactive',
                              active
                                  ? const Color(0xFF3D7B53)
                                  : const Color(0xFF8B8B8B),
                            ),
                          ],
                        ),
                        const SizedBox(height: 4),
                        Text(
                          student['email'],
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.dmSans(
                            fontSize: 10.5,
                            color: OwnerFeaturePalette.textMuted,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(width: 6),
                  const Icon(
                    Icons.chevron_right_rounded,
                    color: OwnerFeaturePalette.textMuted,
                  ),
                ],
              ),
              const SizedBox(height: 15),
              Row(
                children: [
                  Expanded(
                    child: _infoMini(
                      icon: Icons.bed_rounded,
                      label: 'Room',
                      value: student['room'],
                      color: const Color(0xFF3478B7),
                    ),
                  ),
                  Expanded(
                    child: _infoMini(
                      icon: Icons.airline_seat_individual_suite_rounded,
                      label: 'Bed',
                      value: student['bed'],
                      color: const Color(0xFF7D55B4),
                    ),
                  ),
                  Expanded(
                    child: _infoMini(
                      icon: Icons.receipt_long_rounded,
                      label: 'Fee',
                      value: student['feeStatus'],
                      color: feeDue
                          ? const Color(0xFFC07421)
                          : student['feeStatus'] == 'Paid'
                          ? const Color(0xFF3D7B53)
                          : OwnerFeaturePalette.textMuted,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 14),
              if (student['bookingStatus'] == 'pending') ...[
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: () => _issueHostelCredential(student),
                    icon: const Icon(Icons.key_rounded, size: 18),
                    label: const Text('Accept & Generate Hostel Access'),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.teal,
                      foregroundColor: Colors.white,
                      elevation: 0,
                      padding: const EdgeInsets.symmetric(vertical: 12),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(13),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 14),
              ],
              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 12,
                  vertical: 10,
                ),
                decoration: BoxDecoration(
                  color: OwnerFeaturePalette.surfaceSoft,
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Row(
                  children: [
                    Icon(
                      student['parentLinked']
                          ? Icons.family_restroom_rounded
                          : Icons.link_off_rounded,
                      size: 17,
                      color: student['parentLinked']
                          ? AppTheme.teal
                          : OwnerFeaturePalette.textMuted,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        student['parentLinked']
                            ? 'Parent account linked'
                            : 'Parent account not linked',
                        style: GoogleFonts.dmSans(
                          fontSize: 10.5,
                          color: student['parentLinked']
                              ? AppTheme.teal
                              : OwnerFeaturePalette.textMuted,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    if (feeDue)
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        decoration: BoxDecoration(
                          color: const Color(0xFFFFF3E2),
                          borderRadius: BorderRadius.circular(20),
                        ),
                        child: Text(
                          'Fee Due',
                          style: GoogleFonts.dmSans(
                            fontSize: 9,
                            fontWeight: FontWeight.w700,
                            color: const Color(0xFFC07421),
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

  Widget _studentAvatar(String name) {
    final parts = name.trim().split(' ');

    String initials = parts.isEmpty
        ? 'S'
        : parts.first.characters.first.toUpperCase();

    if (parts.length > 1) {
      initials += parts.last.characters.first.toUpperCase();
    }

    return Container(
      width: 50,
      height: 50,
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            AppTheme.teal.withOpacity(0.18),
            AppTheme.tealMint.withOpacity(0.28),
          ],
        ),
        shape: BoxShape.circle,
      ),
      child: Center(
        child: Text(
          initials,
          style: GoogleFonts.dmSans(
            fontSize: 15,
            fontWeight: FontWeight.w800,
            color: AppTheme.teal,
          ),
        ),
      ),
    );
  }

  Widget _infoMini({
    required IconData icon,
    required String label,
    required String value,
    required Color color,
  }) {
    return Column(
      children: [
        Icon(icon, size: 16, color: color),
        const SizedBox(height: 4),
        Text(
          value,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(
            fontSize: 10.5,
            fontWeight: FontWeight.w700,
            color: OwnerFeaturePalette.text,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          label,
          style: GoogleFonts.dmSans(
            fontSize: 9,
            color: OwnerFeaturePalette.textMuted,
          ),
        ),
      ],
    );
  }

  Widget _statusChip(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.10),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        text,
        style: GoogleFonts.dmSans(
          fontSize: 9,
          fontWeight: FontWeight.w700,
          color: color,
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 42),
      decoration: BoxDecoration(
        color: OwnerFeaturePalette.surface,
        borderRadius: BorderRadius.circular(22),
        border: Border.all(color: OwnerFeaturePalette.border),
      ),
      child: Column(
        children: [
          Container(
            width: 70,
            height: 70,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.07),
              shape: BoxShape.circle,
            ),
            child: Icon(Icons.groups_outlined, size: 31, color: AppTheme.teal),
          ),
          const SizedBox(height: 16),
          Text(
            'No students found',
            style: GoogleFonts.playfairDisplay(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: OwnerFeaturePalette.text,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            'Try changing the filters or add a new student.',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              fontSize: 11.5,
              height: 1.5,
              color: OwnerFeaturePalette.textMuted,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFab() {
    return FloatingActionButton.extended(
      backgroundColor: AppTheme.teal,
      foregroundColor: Colors.white,
      elevation: 3,
      icon: const Icon(Icons.person_add_alt_1_rounded),
      label: Text(
        'Add Student',
        style: GoogleFonts.dmSans(fontSize: 12, fontWeight: FontWeight.w700),
      ),
      onPressed: _openAddStudent,
    );
  }

  Future<void> _issueHostelCredential(Map<String, dynamic> student) async {
    final bookingId = int.tryParse('${student['bookingId']}');
    if (bookingId == null) return;
    try {
      final result = await Api().issueOwnerHostelCredential(bookingId);
      if (!mounted) return;
      final password = result['temporary_password']?.toString() ?? '';
      await showDialog<void>(
        context: context,
        // Show inside ResponsiveShell's Navigator rather than the desktop-wide
        // root overlay.
        useRootNavigator: false,
        builder: (context) => AlertDialog(
          title: const Text('Hostel access created'),
          content: SelectableText(
            'Student: ${student['email']}\n\nTemporary hostel password:\n$password\n\nShare this once. The student enters it in My Bookings, then selects their floor, room and bed.',
          ),
          actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('Done'))],
        ),
      );
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Credential generate nahi hua: $error')),
      );
    }
  }

  Future<void> _openStudentDetails(Map<String, dynamic> student) async {
    await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => StudentDetailsScreen(
          student: student,
          hostelName: widget.hostelName,
        ),
      ),
    );
    if (mounted) _loadStudents();
  }

  void _openAddStudent() async {
    final result = await Navigator.push<Map<String, dynamic>>(
      context,
      MaterialPageRoute(
        builder: (_) => AddStudentScreen(hostelName: widget.hostelName),
      ),
    );

    if (result == null) return;

    setState(() {
      _students.insert(0, result);
    });

    if (!mounted) return;

    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: const Color(0xFF285F46),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          content: Row(
            children: [
              const Icon(
                Icons.check_circle_rounded,
                color: Colors.white,
                size: 18,
              ),
              const SizedBox(width: 9),
              Expanded(
                child: Text(
                  '${result['name']} added successfully',
                  style: GoogleFonts.dmSans(color: Colors.white),
                ),
              ),
            ],
          ),
        ),
      );
  }

  void _showMoreMenu() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (_) {
        return SafeArea(
          top: false,
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              color: OwnerFeaturePalette.surfaceStrong,
              borderRadius: BorderRadius.circular(26),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _menuTile(
                  icon: Icons.file_upload_outlined,
                  title: 'Import students',
                  subtitle: 'Upload CSV or Excel file',
                  onTap: () {
                    Navigator.pop(context);
                    _temporaryMessage(
                      'Student import will be connected later.',
                    );
                  },
                ),
                _menuTile(
                  icon: Icons.download_outlined,
                  title: 'Export student list',
                  subtitle: 'Download records',
                  onTap: () {
                    Navigator.pop(context);
                    _exportStudents();
                  },
                ),
                _menuTile(
                  icon: Icons.password_rounded,
                  title: 'Manage credentials',
                  subtitle: 'Student login accounts',
                  onTap: () {
                    Navigator.pop(context);
                    _temporaryMessage(
                      'Credentials screen will be connected next.',
                    );
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _exportStudents() async {
    try {
      final csv = await Api().getOwnerExportCsv('students');
      await Clipboard.setData(ClipboardData(text: csv));
      _temporaryMessage('Student CSV copied to the clipboard.');
    } catch (error) {
      _temporaryMessage(error.toString());
    }
  }

  Widget _menuTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return ListTile(
      onTap: onTap,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
      leading: Container(
        width: 42,
        height: 42,
        decoration: BoxDecoration(
          color: AppTheme.teal.withOpacity(0.08),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Icon(icon, size: 20, color: AppTheme.teal),
      ),
      title: Text(
        title,
        style: GoogleFonts.dmSans(
          fontSize: 13,
          fontWeight: FontWeight.w600,
          color: OwnerFeaturePalette.text,
        ),
      ),
      subtitle: Text(
        subtitle,
        style: GoogleFonts.dmSans(
          fontSize: 10,
          color: OwnerFeaturePalette.textMuted,
        ),
      ),
      trailing: const Icon(Icons.chevron_right_rounded),
    );
  }

  void _temporaryMessage(String message) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          content: Text(message, style: GoogleFonts.dmSans()),
        ),
      );
  }
}
