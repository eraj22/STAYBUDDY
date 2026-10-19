import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme.dart';
import '../../../api.dart';
import '../../../widgets/responsive_shell.dart';
import '../../../widgets/video_background.dart';

class RoomDetailsScreen extends StatefulWidget {
  final String hostelName;
  final Map<String, dynamic> room;

  const RoomDetailsScreen({
    super.key,
    required this.hostelName,
    required this.room,
  });

  @override
  State<RoomDetailsScreen> createState() => _RoomDetailsScreenState();
}

class _RoomDetailsScreenState extends State<RoomDetailsScreen>
    with SingleTickerProviderStateMixin {
  late Map<String, dynamic> _room;

  bool _maintenance = false;

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  final List<Map<String, dynamic>> _students = [
    {
      'name': 'Sara Khan',
      'bed': 'B1',
      'phone': '03111234567',
      'feeStatus': 'Paid',
      'status': 'Active',
    },
    {
      'name': 'Ayesha Ahmed',
      'bed': 'B2',
      'phone': '03331234567',
      'feeStatus': 'Due',
      'status': 'Active',
    },
  ];

  List<String> get _amenities => (_room['amenities'] as List? ?? const [])
      .map((item) => item.toString())
      .toList();

  @override
  void initState() {
    super.initState();

    _room = Map<String, dynamic>.from(widget.room);

    _maintenance = _room['status']?.toString() == 'Maintenance';

    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );

    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);

    _slide = Tween<double>(
      begin: 24,
      end: 0,
    ).animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));

    _anim.forward();
  }

  @override
  void dispose() {
    _anim.dispose();
    super.dispose();
  }

  int get _totalBeds => (_room['totalBeds'] as num?)?.toInt() ?? 0;

  int get _occupiedBeds => (_room['occupiedBeds'] as num?)?.toInt() ?? 0;

  int get _availableBeds => (_totalBeds - _occupiedBeds).clamp(0, 999).toInt();

  double get _occupancy {
    if (_totalBeds == 0) {
      return 0;
    }

    return (_occupiedBeds / _totalBeds).clamp(0.0, 1.0).toDouble();
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
          overlayOpacity: 0.52,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: AnimatedBuilder(
              animation: _anim,
              builder: (context, child) {
                return Transform.translate(
                  offset: Offset(0, _slide.value),
                  child: Opacity(opacity: _fade.value, child: child),
                );
              },
              child: CustomScrollView(
                physics: const BouncingScrollPhysics(),
                slivers: [
                  SliverToBoxAdapter(child: _buildHeader()),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(16, 18, 16, 0),
                    sliver: SliverToBoxAdapter(child: _buildOverviewCard()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(16, 24, 16, 10),
                    sliver: SliverToBoxAdapter(
                      child: _sectionTitle(
                        title: 'Room Information',
                        subtitle: 'Basic room configuration',
                      ),
                    ),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    sliver: SliverToBoxAdapter(child: _buildInformationCard()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(16, 24, 16, 10),
                    sliver: SliverToBoxAdapter(
                      child: _sectionTitle(
                        title: 'Beds',
                        subtitle: 'Configured bed inventory and availability',
                      ),
                    ),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    sliver: SliverToBoxAdapter(child: _buildBedsCard()),
                  ),

                  const SliverToBoxAdapter(child: SizedBox(height: 40)),
                ],
              ),
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
    final status = _maintenance
        ? 'Maintenance'
        : (_room['status']?.toString() ?? 'Available');

    final statusColor = _statusColor(status);

    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _circleButton(
                icon: Icons.arrow_back_rounded,
                onTap: () => Navigator.pop(context),
              ),

              const Spacer(),

              _circleButton(
                icon: Icons.edit_outlined,
                onTap: _showEditRoomSheet,
              ),

              const SizedBox(width: 8),

              _circleButton(
                icon: Icons.more_horiz_rounded,
                onTap: _showMoreMenu,
              ),
            ],
          ),

          const SizedBox(height: 24),

          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  const Color(0xFF073F42).withOpacity(0.95),
                  const Color(0xFF032E31).withOpacity(0.92),
                  Colors.black.withOpacity(0.58),
                ],
              ),
              borderRadius: BorderRadius.circular(28),
              border: Border.all(color: AppTheme.tealMint.withOpacity(0.22)),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.22),
                  blurRadius: 22,
                  offset: const Offset(0, 9),
                ),
              ],
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Container(
                  width: 58,
                  height: 58,
                  decoration: BoxDecoration(
                    color: AppTheme.teal.withOpacity(0.22),
                    borderRadius: BorderRadius.circular(17),
                    border: Border.all(
                      color: AppTheme.tealMint.withOpacity(0.18),
                    ),
                  ),
                  child: const Icon(
                    Icons.meeting_room_rounded,
                    size: 28,
                    color: AppTheme.tealMint,
                  ),
                ),

                const SizedBox(width: 13),

                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Room ${_room['roomNo'] ?? '-'}',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.playfairDisplay(
                          fontSize: 26,
                          height: 1,
                          fontWeight: FontWeight.w700,
                          color: Colors.white,
                        ),
                      ),

                      const SizedBox(height: 7),

                      Text(
                        '${_room['floor'] ?? '-'} · ${_room['type'] ?? '-'}',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          fontSize: 9.5,
                          color: Colors.white.withOpacity(0.56),
                        ),
                      ),

                      const SizedBox(height: 3),

                      Text(
                        widget.hostelName,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          fontSize: 8.5,
                          color: Colors.white.withOpacity(0.40),
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(width: 8),

                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 9,
                    vertical: 5,
                  ),
                  decoration: BoxDecoration(
                    color: statusColor.withOpacity(0.16),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: statusColor.withOpacity(0.28)),
                  ),
                  child: Text(
                    status,
                    style: GoogleFonts.dmSans(
                      fontSize: 8,
                      fontWeight: FontWeight.w800,
                      color: statusColor,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
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
            color: Colors.black.withOpacity(0.28),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white.withOpacity(0.16)),
          ),
          child: Icon(icon, size: 20, color: Colors.white),
        ),
      ),
    );
  }

  // ===========================================================================
  // OVERVIEW
  // ===========================================================================

  Widget _buildOverviewCard() {
    return _glassCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Occupancy Overview',
                      style: GoogleFonts.dmSans(
                        fontSize: 13,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                      ),
                    ),

                    const SizedBox(height: 3),

                    Text(
                      'Current bed utilization',
                      style: GoogleFonts.dmSans(
                        fontSize: 8.5,
                        color: Colors.white.withOpacity(0.44),
                      ),
                    ),
                  ],
                ),
              ),

              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 10,
                  vertical: 6,
                ),
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.17),
                  borderRadius: BorderRadius.circular(18),
                ),
                child: Text(
                  '${(_occupancy * 100).round()}%',
                  style: GoogleFonts.dmSans(
                    fontSize: 10,
                    fontWeight: FontWeight.w800,
                    color: AppTheme.tealMint,
                  ),
                ),
              ),
            ],
          ),

          const SizedBox(height: 17),

          ClipRRect(
            borderRadius: BorderRadius.circular(20),
            child: LinearProgressIndicator(
              value: _occupancy,
              minHeight: 9,
              backgroundColor: Colors.white.withOpacity(0.08),
              valueColor: const AlwaysStoppedAnimation<Color>(AppTheme.teal),
            ),
          ),

          const SizedBox(height: 18),

          Row(
            children: [
              Expanded(
                child: _overviewItem(
                  value: '$_totalBeds',
                  label: 'Total Beds',
                  icon: Icons.bed_rounded,
                ),
              ),

              _verticalDivider(),

              Expanded(
                child: _overviewItem(
                  value: '$_occupiedBeds',
                  label: 'Occupied',
                  icon: Icons.people_rounded,
                ),
              ),

              _verticalDivider(),

              Expanded(
                child: _overviewItem(
                  value: '$_availableBeds',
                  label: 'Available',
                  icon: Icons.event_available_rounded,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _overviewItem({
    required String value,
    required String label,
    required IconData icon,
  }) {
    return Column(
      children: [
        Container(
          width: 37,
          height: 37,
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.16),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, size: 17, color: AppTheme.tealMint),
        ),

        const SizedBox(height: 7),

        Text(
          value,
          style: GoogleFonts.dmSans(
            fontSize: 17,
            fontWeight: FontWeight.w800,
            color: Colors.white,
          ),
        ),

        const SizedBox(height: 2),

        Text(
          label,
          textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(
            fontSize: 8.5,
            color: Colors.white.withOpacity(0.43),
          ),
        ),
      ],
    );
  }

  Widget _verticalDivider() {
    return Container(
      width: 1,
      height: 48,
      color: Colors.white.withOpacity(0.08),
    );
  }

  // ===========================================================================
  // SECTION TITLE
  // ===========================================================================

  Widget _sectionTitle({required String title, required String subtitle}) {
    return Row(
      children: [
        Container(
          width: 4,
          height: 31,
          decoration: BoxDecoration(
            color: AppTheme.teal,
            borderRadius: BorderRadius.circular(3),
          ),
        ),

        const SizedBox(width: 10),

        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: GoogleFonts.dmSans(
                  fontSize: 13.5,
                  fontWeight: FontWeight.w800,
                  color: Colors.white,
                ),
              ),

              const SizedBox(height: 2),

              Text(
                subtitle,
                style: GoogleFonts.dmSans(
                  fontSize: 8.8,
                  color: Colors.white.withOpacity(0.50),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // INFORMATION
  // ===========================================================================

  Widget _buildInformationCard() {
    return _glassCard(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 3),
      child: Column(
        children: [
          _infoRow(
            icon: Icons.meeting_room_outlined,
            label: 'Room Number',
            value: _room['roomNo']?.toString() ?? '-',
          ),

          _divider(),

          _infoRow(
            icon: Icons.layers_outlined,
            label: 'Floor',
            value: _room['floor']?.toString() ?? '-',
          ),

          _divider(),

          _infoRow(
            icon: Icons.bed_outlined,
            label: 'Room Type',
            value: _room['type']?.toString() ?? '-',
          ),

          _divider(),

          _infoRow(
            icon: Icons.payments_outlined,
            label: 'Monthly Rent',
            value:
                'PKR ${_formatAmount((_room['rent'] as num?)?.toInt() ?? 0)}',
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // BEDS
  // ===========================================================================

  Widget _buildBedsCard() {
    if (_totalBeds <= 0) {
      return _glassCard(
        child: Center(
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 20),
            child: Text(
              'No beds configured for this room.',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                fontSize: 10,
                color: Colors.white.withOpacity(0.48),
              ),
            ),
          ),
        ),
      );
    }

    return _glassCard(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 3),
      child: Column(
        children: List.generate(_totalBeds, (index) {
          final configuredBeds = (_room['beds'] as List? ?? const [])
              .map((bed) => Map<String, dynamic>.from(bed as Map))
              .toList();
          final bed = index < configuredBeds.length
              ? configuredBeds[index]
              : const <String, dynamic>{};
          final bedNo = bed['number']?.toString() ?? '${index + 1}';
          final occupied = bed['occupied'] == true;

          return Column(
            children: [
              _bedTile(bedNo: bedNo, occupied: occupied),

              if (index < _totalBeds - 1) _divider(),
            ],
          );
        }),
      ),
    );
  }

  Widget _bedTile({required String bedNo, required bool occupied}) {
    final color = occupied ? const Color(0xFF74B7FF) : const Color(0xFF66D8A6);

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 11),
      child: Row(
        children: [
          Container(
            width: 41,
            height: 41,
            decoration: BoxDecoration(
              color: color.withOpacity(0.12),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(Icons.bed_rounded, size: 19, color: color),
          ),

          const SizedBox(width: 11),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Bed $bedNo',
                  style: GoogleFonts.dmSans(
                    fontSize: 11.5,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),

                const SizedBox(height: 3),

                Text(
                  occupied ? 'Occupied' : 'Empty / available',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 8.5,
                    color: Colors.white.withOpacity(0.44),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(width: 8),

          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: color.withOpacity(0.12),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Text(
              occupied ? 'Occupied' : 'Available',
              style: GoogleFonts.dmSans(
                fontSize: 7.6,
                fontWeight: FontWeight.w800,
                color: color,
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // RESIDENTS
  // ===========================================================================

  Widget _buildResidents() {
    if (_occupiedBeds == 0 || _students.isEmpty) {
      return _emptyResidents();
    }

    final visibleStudents = _students.take(_occupiedBeds).toList();

    return _glassCard(
      padding: EdgeInsets.zero,
      child: Column(
        children: List.generate(visibleStudents.length, (index) {
          final student = visibleStudents[index];

          return Column(
            children: [
              _residentTile(student),

              if (index < visibleStudents.length - 1)
                Padding(
                  padding: const EdgeInsets.only(left: 65, right: 14),
                  child: _divider(),
                ),
            ],
          );
        }),
      ),
    );
  }

  Widget _residentTile(Map<String, dynamic> student) {
    final feeDue = student['feeStatus'] == 'Due';

    return Padding(
      padding: const EdgeInsets.all(14),
      child: Row(
        children: [
          _studentAvatar(student['name'].toString()),

          const SizedBox(width: 11),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  student['name'].toString(),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 11.5,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),

                const SizedBox(height: 3),

                Text(
                  'Bed ${student['bed']} · ${student['phone']}',
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 8.5,
                    color: Colors.white.withOpacity(0.43),
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(width: 8),

          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: feeDue
                  ? const Color(0xFFFFB05F).withOpacity(0.12)
                  : const Color(0xFF66D8A6).withOpacity(0.12),
              borderRadius: BorderRadius.circular(16),
            ),
            child: Text(
              feeDue ? 'Fee Due' : 'Paid',
              style: GoogleFonts.dmSans(
                fontSize: 7.8,
                fontWeight: FontWeight.w800,
                color: feeDue
                    ? const Color(0xFFFFB05F)
                    : const Color(0xFF66D8A6),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _studentAvatar(String name) {
    final parts = name.trim().split(' ');

    String initials = parts.isNotEmpty && parts.first.isNotEmpty
        ? parts.first[0].toUpperCase()
        : 'S';

    if (parts.length > 1 && parts.last.isNotEmpty) {
      initials += parts.last[0].toUpperCase();
    }

    return Container(
      width: 40,
      height: 40,
      decoration: BoxDecoration(
        color: AppTheme.teal.withOpacity(0.18),
        shape: BoxShape.circle,
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.18)),
      ),
      child: Center(
        child: Text(
          initials,
          style: GoogleFonts.dmSans(
            fontSize: 11,
            fontWeight: FontWeight.w800,
            color: AppTheme.tealMint,
          ),
        ),
      ),
    );
  }

  Widget _emptyResidents() {
    return _glassCard(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 23),
        child: Column(
          children: [
            Container(
              width: 57,
              height: 57,
              decoration: BoxDecoration(
                color: AppTheme.teal.withOpacity(0.15),
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.person_off_outlined,
                size: 25,
                color: AppTheme.tealMint,
              ),
            ),

            const SizedBox(height: 12),

            Text(
              'No students assigned',
              style: GoogleFonts.dmSans(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),

            const SizedBox(height: 4),

            Text(
              'Available beds can be assigned from this room.',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                fontSize: 8.8,
                color: Colors.white.withOpacity(0.45),
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // AMENITIES
  // ===========================================================================

  Widget _buildAmenities() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Align(
          alignment: Alignment.centerRight,
          child: TextButton.icon(
            onPressed: _editAmenities,
            icon: const Icon(Icons.edit_outlined, size: 16),
            label: const Text('Edit amenities'),
            style: TextButton.styleFrom(foregroundColor: AppTheme.tealMint),
          ),
        ),
        if (_amenities.isEmpty)
          _glassCard(
            child: Text(
              'No amenities added yet. Tap Edit amenities to add them.',
              style: GoogleFonts.dmSans(
                fontSize: 10,
                color: Colors.white.withOpacity(0.60),
              ),
            ),
          )
        else
          LayoutBuilder(
            builder: (context, constraints) {
              final columns = constraints.maxWidth < 310 ? 1 : 2;

              const gap = 10.0;

              final width = columns == 1
                  ? constraints.maxWidth
                  : (constraints.maxWidth - gap) / 2;

              return Wrap(
                spacing: gap,
                runSpacing: gap,
                children: _amenities.map((amenity) {
                  return SizedBox(
                    width: width,
                    child: Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: const Color(0xD909292B),
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(
                          color: Colors.white.withOpacity(0.09),
                        ),
                      ),
                      child: Row(
                        children: [
                          Container(
                            width: 36,
                            height: 36,
                            decoration: BoxDecoration(
                              color: AppTheme.teal.withOpacity(0.16),
                              borderRadius: BorderRadius.circular(11),
                            ),
                            child: Icon(
                              _amenityIcon(amenity),
                              size: 17,
                              color: AppTheme.tealMint,
                            ),
                          ),

                          const SizedBox(width: 9),

                          Expanded(
                            child: Text(
                              amenity,
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                              style: GoogleFonts.dmSans(
                                fontSize: 9.5,
                                fontWeight: FontWeight.w600,
                                color: Colors.white,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                }).toList(),
              );
            },
          ),
      ],
    );
  }

  Future<void> _editAmenities() async {
    const options = [
      'WiFi',
      'Attached Bath',
      'Wardrobe',
      'Study Table',
      'Air Conditioning',
      'Balcony',
    ];
    final selected = _amenities.toSet();
    final result = await showModalBottomSheet<List<String>>(
      context: context,
      isScrollControlled: true,
      backgroundColor: const Color(0xFF082C2F),
      builder: (sheetContext) => StatefulBuilder(
        builder: (context, setSheetState) => SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Room amenities',
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Select facilities available in this room.',
                  style: GoogleFonts.dmSans(
                    color: Colors.white70,
                    fontSize: 11,
                  ),
                ),
                const SizedBox(height: 12),
                ...options.map(
                  (option) => CheckboxListTile(
                    value: selected.contains(option),
                    onChanged: (value) => setSheetState(() {
                      value == true
                          ? selected.add(option)
                          : selected.remove(option);
                    }),
                    activeColor: AppTheme.teal,
                    checkColor: Colors.white,
                    title: Text(
                      option,
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    contentPadding: EdgeInsets.zero,
                  ),
                ),
                const SizedBox(height: 8),
                SizedBox(
                  width: double.infinity,
                  child: ElevatedButton(
                    onPressed: () =>
                        Navigator.pop(sheetContext, selected.toList()),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppTheme.teal,
                      foregroundColor: Colors.white,
                    ),
                    child: const Text('Save amenities'),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
    if (result == null) return;
    try {
      await Api().updateRoom(
        roomId: (_room['id'] as num).toInt(),
        roomNumber: _room['roomNo']?.toString() ?? '',
        floorLabel: _room['floor']?.toString(),
        capacity: _totalBeds,
        availableCapacity: _availableBeds,
        roomType: _room['type']?.toString() ?? 'Shared',
        monthlyRent: (_room['rent'] as num?)?.toInt() ?? 0,
        amenities: result,
        notes: _room['notes']?.toString(),
      );
      if (!mounted) return;
      setState(() => _room['amenities'] = result);
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not save amenities: $error')),
      );
    }
  }

  IconData _amenityIcon(String amenity) {
    switch (amenity) {
      case 'WiFi':
        return Icons.wifi_rounded;

      case 'Attached Bath':
        return Icons.bathtub_outlined;

      case 'Wardrobe':
        return Icons.door_sliding_outlined;

      case 'Study Table':
        return Icons.desk_outlined;

      default:
        return Icons.check_circle_outline_rounded;
    }
  }

  // ===========================================================================
  // ROOM CONTROLS
  // ===========================================================================

  Widget _buildRoomControls() {
    return Column(
      children: [
        _glassCard(
          child: Row(
            children: [
              Container(
                width: 43,
                height: 43,
                decoration: BoxDecoration(
                  color: const Color(0xFFFFB05F).withOpacity(0.14),
                  borderRadius: BorderRadius.circular(13),
                ),
                child: const Icon(
                  Icons.build_outlined,
                  color: Color(0xFFFFB05F),
                  size: 20,
                ),
              ),

              const SizedBox(width: 11),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Maintenance Mode',
                      style: GoogleFonts.dmSans(
                        fontSize: 11.5,
                        fontWeight: FontWeight.w700,
                        color: Colors.white,
                      ),
                    ),

                    const SizedBox(height: 3),

                    Text(
                      'Prevent new bed assignments while maintenance is active.',
                      style: GoogleFonts.dmSans(
                        fontSize: 8.3,
                        height: 1.35,
                        color: Colors.white.withOpacity(0.45),
                      ),
                    ),
                  ],
                ),
              ),

              Transform.scale(
                scale: 0.88,
                child: Switch.adaptive(
                  value: _maintenance,
                  activeColor: AppTheme.tealMint,
                  activeTrackColor: AppTheme.teal,
                  onChanged: _toggleMaintenance,
                ),
              ),
            ],
          ),
        ),

        const SizedBox(height: 12),

        SizedBox(
          width: double.infinity,
          height: 50,
          child: ElevatedButton.icon(
            onPressed: _showEditRoomSheet,
            icon: const Icon(Icons.edit_outlined, size: 18),
            label: Text(
              'Edit Room Information',
              style: GoogleFonts.dmSans(
                fontSize: 11.5,
                fontWeight: FontWeight.w800,
              ),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.teal,
              foregroundColor: Colors.white,
              elevation: 0,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
            ),
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // COMMON
  // ===========================================================================

  Widget _glassCard({
    required Widget child,
    EdgeInsetsGeometry padding = const EdgeInsets.all(16),
  }) {
    return Container(
      width: double.infinity,
      padding: padding,
      decoration: BoxDecoration(
        color: const Color(0xD909292B),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.09)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.10),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: child,
    );
  }

  Widget _infoRow({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12),
      child: Row(
        children: [
          Container(
            width: 39,
            height: 39,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.16),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(icon, size: 18, color: AppTheme.tealMint),
          ),

          const SizedBox(width: 11),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: GoogleFonts.dmSans(
                    fontSize: 8.2,
                    color: Colors.white.withOpacity(0.42),
                  ),
                ),

                const SizedBox(height: 3),

                Text(
                  value,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    fontSize: 11,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _divider() {
    return Divider(
      height: 1,
      thickness: 0.7,
      color: Colors.white.withOpacity(0.07),
    );
  }

  // ===========================================================================
  // BED STUDENT SHEET
  // ===========================================================================

  void _showBedStudentSheet(String bedNo, Map<String, dynamic>? student) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (sheetContext) {
        return SafeArea(
          top: false,
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.fromLTRB(18, 12, 18, 20),
            decoration: BoxDecoration(
              color: const Color(0xFF082C2F),
              borderRadius: BorderRadius.circular(26),
              border: Border.all(color: AppTheme.tealMint.withOpacity(0.17)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _sheetHandle(),

                Text(
                  'Bed $bedNo',
                  style: GoogleFonts.playfairDisplay(
                    fontSize: 22,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),

                const SizedBox(height: 5),

                Text(
                  student?['name']?.toString() ?? 'Assigned student',
                  style: GoogleFonts.dmSans(
                    fontSize: 9.5,
                    color: Colors.white.withOpacity(0.48),
                  ),
                ),

                const SizedBox(height: 16),

                _darkInfoRow(
                  icon: Icons.person_outline_rounded,
                  label: 'Student',
                  value: student?['name']?.toString() ?? '-',
                ),

                const SizedBox(height: 8),

                _darkInfoRow(
                  icon: Icons.phone_outlined,
                  label: 'Phone',
                  value: student?['phone']?.toString() ?? '-',
                ),

                const SizedBox(height: 15),

                SizedBox(
                  width: double.infinity,
                  height: 47,
                  child: OutlinedButton.icon(
                    onPressed: () {
                      Navigator.pop(sheetContext);

                      _showMessage(
                        'Move student flow will be connected later.',
                      );
                    },
                    icon: const Icon(Icons.swap_horiz_rounded, size: 17),
                    label: Text(
                      'Move Student',
                      style: GoogleFonts.dmSans(fontWeight: FontWeight.w700),
                    ),
                    style: OutlinedButton.styleFrom(
                      foregroundColor: AppTheme.tealMint,
                      side: BorderSide(
                        color: AppTheme.tealMint.withOpacity(0.40),
                      ),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  // ===========================================================================
  // ASSIGN BED
  // ===========================================================================

  void _showAssignBedSheet(String bedNo) {
    final candidates = ['Amna Raza', 'Noor Fatima', 'Mehak Ali'];

    String? selectedStudent;

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (sheetContext) {
        return StatefulBuilder(
          builder: (context, setSheetState) {
            return SafeArea(
              top: false,
              child: Container(
                margin: const EdgeInsets.all(12),
                padding: const EdgeInsets.fromLTRB(18, 12, 18, 20),
                decoration: BoxDecoration(
                  color: const Color(0xFF082C2F),
                  borderRadius: BorderRadius.circular(26),
                  border: Border.all(
                    color: AppTheme.tealMint.withOpacity(0.17),
                  ),
                ),
                child: SingleChildScrollView(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _sheetHandle(),

                      Text(
                        'Assign Bed $bedNo',
                        style: GoogleFonts.playfairDisplay(
                          fontSize: 22,
                          fontWeight: FontWeight.w700,
                          color: Colors.white,
                        ),
                      ),

                      const SizedBox(height: 5),

                      Text(
                        'Select an unassigned student.',
                        style: GoogleFonts.dmSans(
                          fontSize: 9.5,
                          color: Colors.white.withOpacity(0.47),
                        ),
                      ),

                      const SizedBox(height: 15),

                      ...candidates.map((name) {
                        final selected = selectedStudent == name;

                        return Padding(
                          padding: const EdgeInsets.only(bottom: 9),
                          child: InkWell(
                            borderRadius: BorderRadius.circular(15),
                            onTap: () {
                              setSheetState(() {
                                selectedStudent = name;
                              });
                            },
                            child: Container(
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: selected
                                    ? AppTheme.teal.withOpacity(0.16)
                                    : Colors.black.withOpacity(0.15),
                                borderRadius: BorderRadius.circular(15),
                                border: Border.all(
                                  color: selected
                                      ? AppTheme.tealMint.withOpacity(0.38)
                                      : Colors.white.withOpacity(0.07),
                                ),
                              ),
                              child: Row(
                                children: [
                                  _studentAvatar(name),

                                  const SizedBox(width: 10),

                                  Expanded(
                                    child: Text(
                                      name,
                                      style: GoogleFonts.dmSans(
                                        fontSize: 11,
                                        fontWeight: FontWeight.w700,
                                        color: Colors.white,
                                      ),
                                    ),
                                  ),

                                  Icon(
                                    selected
                                        ? Icons.radio_button_checked_rounded
                                        : Icons.radio_button_off_rounded,
                                    color: selected
                                        ? AppTheme.tealMint
                                        : Colors.white.withOpacity(0.35),
                                  ),
                                ],
                              ),
                            ),
                          ),
                        );
                      }),

                      const SizedBox(height: 9),

                      SizedBox(
                        width: double.infinity,
                        height: 48,
                        child: ElevatedButton(
                          onPressed: selectedStudent == null
                              ? null
                              : () {
                                  Navigator.pop(sheetContext);

                                  setState(() {
                                    _room['occupiedBeds'] = (_occupiedBeds + 1)
                                        .clamp(0, _totalBeds)
                                        .toInt();

                                    _room['status'] = _availableBeds > 1
                                        ? 'Available'
                                        : 'Full';
                                  });

                                  _showMessage(
                                    '$selectedStudent assigned to Bed $bedNo.',
                                  );
                                },
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppTheme.teal,
                            foregroundColor: Colors.white,
                            disabledBackgroundColor: AppTheme.teal.withOpacity(
                              0.35,
                            ),
                            elevation: 0,
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(15),
                            ),
                          ),
                          child: Text(
                            'Assign Student',
                            style: GoogleFonts.dmSans(
                              fontWeight: FontWeight.w800,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            );
          },
        );
      },
    );
  }

  // ===========================================================================
  // EDIT ROOM
  // ===========================================================================

  void _showEditRoomSheet() {
    final rentCtrl = TextEditingController(
      text: (_room['rent'] ?? 0).toString(),
    );

    String selectedType = _room['type']?.toString() ?? 'Double Sharing';

    final roomTypes = [
      'Single Room',
      'Double Sharing',
      'Triple Sharing',
      'Four Sharing',
    ];

    if (!roomTypes.contains(selectedType)) {
      selectedType = 'Double Sharing';
    }

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (sheetContext) {
        return StatefulBuilder(
          builder: (context, setSheetState) {
            return Padding(
              padding: EdgeInsets.only(
                bottom: MediaQuery.of(context).viewInsets.bottom,
              ),
              child: SafeArea(
                top: false,
                child: Container(
                  margin: const EdgeInsets.all(12),
                  padding: const EdgeInsets.fromLTRB(18, 12, 18, 20),
                  decoration: BoxDecoration(
                    color: const Color(0xFF082C2F),
                    borderRadius: BorderRadius.circular(26),
                    border: Border.all(
                      color: AppTheme.tealMint.withOpacity(0.17),
                    ),
                  ),
                  child: SingleChildScrollView(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        _sheetHandle(),

                        Text(
                          'Edit Room ${_room['roomNo']}',
                          style: GoogleFonts.playfairDisplay(
                            fontSize: 22,
                            fontWeight: FontWeight.w700,
                            color: Colors.white,
                          ),
                        ),

                        const SizedBox(height: 17),

                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 13),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(15),
                          ),
                          child: DropdownButtonHideUnderline(
                            child: DropdownButton<String>(
                              value: selectedType,
                              isExpanded: true,
                              dropdownColor: Colors.white,
                              iconEnabledColor: const Color(0xFF0A6668),
                              borderRadius: BorderRadius.circular(16),
                              items: roomTypes.map((item) {
                                return DropdownMenuItem<String>(
                                  value: item,
                                  child: Text(
                                    item,
                                    style: GoogleFonts.dmSans(
                                      fontSize: 12,
                                      color: const Color(0xFF143637),
                                    ),
                                  ),
                                );
                              }).toList(),
                              onChanged: (value) {
                                if (value == null) {
                                  return;
                                }

                                setSheetState(() {
                                  selectedType = value;
                                });
                              },
                            ),
                          ),
                        ),

                        const SizedBox(height: 12),

                        _whiteTextField(
                          controller: rentCtrl,
                          hint: 'Monthly rent',
                          icon: Icons.payments_outlined,
                        ),

                        const SizedBox(height: 18),

                        SizedBox(
                          width: double.infinity,
                          height: 48,
                          child: ElevatedButton(
                            onPressed: () async {
                              final rent =
                                  int.tryParse(rentCtrl.text.trim()) ?? 0;

                              if (rent <= 0) {
                                _showMessage('Enter a valid monthly rent.');
                                return;
                              }

                              final roomId = int.tryParse(
                                _room['id']?.toString() ?? '',
                              );

                              final capacity =
                                  int.tryParse(
                                    _room['totalBeds']?.toString() ?? '0',
                                  ) ??
                                  0;

                              final occupied =
                                  int.tryParse(
                                    _room['occupiedBeds']?.toString() ?? '0',
                                  ) ??
                                  0;

                              final available = (capacity - occupied).clamp(
                                0,
                                capacity,
                              );

                              if (roomId == null) {
                                _showMessage('Invalid room id.');
                                return;
                              }

                              try {
                                final response = await Api().updateRoom(
                                  roomId: roomId,
                                  roomNumber: _room['roomNo']?.toString() ?? '',
                                  capacity: capacity,
                                  availableCapacity: available,
                                  roomType: selectedType,
                                  monthlyRent: rent,
                                );
                                if (!mounted) return;

                                final updated = Map<String, dynamic>.from(
                                  response['room'] as Map,
                                );

                                setState(() {
                                  _room['type'] =
                                      updated['room_type']?.toString() ??
                                      selectedType;

                                  _room['rent'] = rent;

                                  _room['totalBeds'] =
                                      int.tryParse(
                                        updated['capacity']?.toString() ?? '0',
                                      ) ??
                                      capacity;

                                  final updatedAvailable =
                                      int.tryParse(
                                        updated['available_capacity']
                                                ?.toString() ??
                                            '0',
                                      ) ??
                                      available;

                                  _room['occupiedBeds'] =
                                      (_room['totalBeds'] - updatedAvailable)
                                          .clamp(0, _room['totalBeds']);
                                });

                                Navigator.pop(sheetContext);

                                _showMessage('Room updated successfully.');
                              } catch (e) {
                                if (!mounted) return;

                                _showMessage('Could not update room: $e');
                              }
                            },
                            style: ElevatedButton.styleFrom(
                              backgroundColor: AppTheme.teal,
                              foregroundColor: Colors.white,
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(15),
                              ),
                            ),
                            child: Text(
                              'Save Changes',
                              style: GoogleFonts.dmSans(
                                fontWeight: FontWeight.w800,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            );
          },
        );
      },
    ).whenComplete(() {
      rentCtrl.dispose();
    });
  }

  Widget _whiteTextField({
    required TextEditingController controller,
    required String hint,
    required IconData icon,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(15),
      ),
      child: TextField(
        controller: controller,
        keyboardType: TextInputType.number,
        style: GoogleFonts.dmSans(
          fontSize: 13,
          fontWeight: FontWeight.w500,
          color: const Color(0xFF143637),
        ),
        decoration: InputDecoration(
          hintText: hint,
          hintStyle: GoogleFonts.dmSans(
            fontSize: 12,
            color: const Color(0xFF879596),
          ),
          prefixIcon: Icon(icon, size: 19, color: const Color(0xFF0A6668)),
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(
            horizontal: 14,
            vertical: 15,
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // MAINTENANCE
  // ===========================================================================

  void _toggleMaintenance(bool value) {
    if (value) {
      showDialog(
        context: context,
        builder: (dialogContext) {
          return AlertDialog(
            backgroundColor: const Color(0xFF082C2F),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(22),
            ),
            title: Text(
              'Enable Maintenance?',
              style: GoogleFonts.playfairDisplay(
                fontSize: 20,
                fontWeight: FontWeight.w700,
                color: Colors.white,
              ),
            ),
            content: Text(
              'New students should not be assigned to this room while maintenance is active.',
              style: GoogleFonts.dmSans(
                fontSize: 10.5,
                height: 1.5,
                color: Colors.white.withOpacity(0.56),
              ),
            ),
            actions: [
              TextButton(
                onPressed: () {
                  Navigator.pop(dialogContext);
                },
                child: Text(
                  'Cancel',
                  style: GoogleFonts.dmSans(
                    color: Colors.white.withOpacity(0.60),
                  ),
                ),
              ),

              ElevatedButton(
                onPressed: () {
                  Navigator.pop(dialogContext);

                  setState(() {
                    _maintenance = true;

                    _room['status'] = 'Maintenance';
                  });
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFB36F25),
                  foregroundColor: Colors.white,
                ),
                child: const Text('Enable'),
              ),
            ],
          );
        },
      );
    } else {
      setState(() {
        _maintenance = false;

        _room['status'] = _availableBeds > 0 ? 'Available' : 'Full';
      });
    }
  }

  // ===========================================================================
  // MORE MENU
  // ===========================================================================

  void _showMoreMenu() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (sheetContext) {
        return SafeArea(
          top: false,
          child: Container(
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 16),
            decoration: BoxDecoration(
              color: const Color(0xFF082C2F),
              borderRadius: BorderRadius.circular(26),
              border: Border.all(color: AppTheme.tealMint.withOpacity(0.17)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                _sheetHandle(),

                _menuTile(
                  icon: Icons.delete_outline_rounded,
                  title: 'Delete Room',
                  color: const Color(0xFFFF7777),
                  onTap: () {
                    Navigator.pop(sheetContext);

                    _confirmDelete();
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _menuTile({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
    Color? color,
  }) {
    final itemColor = color ?? AppTheme.tealMint;

    return Padding(
      padding: const EdgeInsets.only(bottom: 7),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(15),
        child: Container(
          padding: const EdgeInsets.all(11),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.15),
            borderRadius: BorderRadius.circular(15),
            border: Border.all(color: Colors.white.withOpacity(0.07)),
          ),
          child: Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: itemColor.withOpacity(0.13),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: itemColor, size: 19),
              ),

              const SizedBox(width: 11),

              Expanded(
                child: Text(
                  title,
                  style: GoogleFonts.dmSans(
                    fontSize: 11.5,
                    fontWeight: FontWeight.w700,
                    color: color ?? Colors.white,
                  ),
                ),
              ),

              Icon(Icons.chevron_right_rounded, color: itemColor, size: 18),
            ],
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // DELETE
  // ===========================================================================

  void _confirmDelete() {
    showDialog(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          backgroundColor: const Color(0xFF082C2F),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(22),
          ),
          title: Text(
            'Delete Room ${_room['roomNo']}?',
            style: GoogleFonts.playfairDisplay(
              fontSize: 20,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),
          content: Text(
            _occupiedBeds > 0
                ? 'This room currently has assigned students. Move them before deleting the room.'
                : 'This room will be removed from the hostel.',
            style: GoogleFonts.dmSans(
              fontSize: 10.5,
              height: 1.5,
              color: Colors.white.withOpacity(0.56),
            ),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(dialogContext),
              child: Text(
                'Cancel',
                style: GoogleFonts.dmSans(
                  color: Colors.white.withOpacity(0.60),
                ),
              ),
            ),

            ElevatedButton(
              onPressed: _occupiedBeds > 0
                  ? null
                  : () {
                      Navigator.pop(dialogContext);

                      Navigator.pop(context, {
                        'deleted': true,
                        'roomNo': _room['roomNo'],
                      });
                    },
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFC94B4B),
                foregroundColor: Colors.white,
              ),
              child: const Text('Delete'),
            ),
          ],
        );
      },
    );
  }

  // ===========================================================================
  // SHEET HELPERS
  // ===========================================================================

  Widget _sheetHandle() {
    return Center(
      child: Container(
        width: 40,
        height: 4,
        margin: const EdgeInsets.only(bottom: 17),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.18),
          borderRadius: BorderRadius.circular(3),
        ),
      ),
    );
  }

  Widget _darkInfoRow({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(11),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.15),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.07)),
      ),
      child: Row(
        children: [
          Container(
            width: 37,
            height: 37,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.16),
              borderRadius: BorderRadius.circular(11),
            ),
            child: Icon(icon, size: 17, color: AppTheme.tealMint),
          ),

          const SizedBox(width: 10),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  label,
                  style: GoogleFonts.dmSans(
                    fontSize: 8,
                    color: Colors.white.withOpacity(0.42),
                  ),
                ),

                const SizedBox(height: 2),

                Text(
                  value,
                  style: GoogleFonts.dmSans(
                    fontSize: 10.5,
                    fontWeight: FontWeight.w700,
                    color: Colors.white,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // STATUS
  // ===========================================================================

  Color _statusColor(String status) {
    switch (status) {
      case 'Available':
        return const Color(0xFF66D8A6);

      case 'Full':
        return const Color(0xFFFFB05F);

      case 'Maintenance':
        return const Color(0xFFFF7777);

      default:
        return AppTheme.tealMint;
    }
  }

  // ===========================================================================
  // FORMAT
  // ===========================================================================

  String _formatAmount(int amount) {
    final value = amount.toString();

    final buffer = StringBuffer();

    for (int i = 0; i < value.length; i++) {
      final remaining = value.length - i;

      buffer.write(value[i]);

      if (remaining > 1 && remaining % 3 == 1) {
        buffer.write(',');
      }
    }

    return buffer.toString();
  }

  // ===========================================================================
  // MESSAGE
  // ===========================================================================

  void _showMessage(String message) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          margin: const EdgeInsets.all(16),
          backgroundColor: const Color(0xFF0A575A),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(15),
          ),
          content: Text(
            message,
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 10.5,
              fontWeight: FontWeight.w600,
            ),
          ),
        ),
      );
  }
}
