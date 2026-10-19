import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';
import '../../../api.dart';
import '../../../theme.dart';
import '../../../widgets/responsive_shell.dart';
import '../../../widgets/video_background.dart';

import 'room_details_screen.dart';
import 'add_room_screen.dart';

class OwnerRoomsScreen extends StatefulWidget {
  final String hostelName;
  final VoidCallback? onBackToDashboard;

  const OwnerRoomsScreen({
    super.key,
    required this.hostelName,
    this.onBackToDashboard,
  });

  @override
  State<OwnerRoomsScreen> createState() => _OwnerRoomsScreenState();
}

class _OwnerRoomsScreenState extends State<OwnerRoomsScreen>
    with SingleTickerProviderStateMixin {
  final TextEditingController _searchCtrl = TextEditingController();

  String _selectedFilter = 'All';
  String _selectedFloor = 'All Floors';

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  final List<String> _filters = ['All', 'Available', 'Full'];

  final List<String> _floors = ['All Floors'];

  final List<Map<String, dynamic>> _rooms = [];
  bool _loadingRooms = false;

  @override
  void initState() {
    super.initState();

    _anim = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 700),
    );

    _fade = CurvedAnimation(parent: _anim, curve: Curves.easeOut);

    _slide = Tween<double>(
      begin: 22,
      end: 0,
    ).animate(CurvedAnimation(parent: _anim, curve: Curves.easeOutCubic));

    _anim.forward();
    _loadRooms();
  }

  Future<void> _loadRooms() async {
    if (_loadingRooms) return;

    setState(() {
      _loadingRooms = true;
    });

    try {
      final hostels = await Api().getOwnerHostels();

      if (hostels.isEmpty) {
        throw Exception('No hostel found');
      }

      Map<String, dynamic>? currentHostel;

      for (final item in hostels) {
        if (item is Map) {
          final data = Map<String, dynamic>.from(item);

          if (data['name']?.toString().trim().toLowerCase() ==
              widget.hostelName.trim().toLowerCase()) {
            currentHostel = data;
            break;
          }
        }
      }

      currentHostel ??= Map<String, dynamic>.from(hostels.first as Map);

      final hostelId = int.tryParse(currentHostel['id']?.toString() ?? '');

      if (hostelId == null) {
        throw Exception('Invalid hostel id');
      }

      final results = await Future.wait([
        Api().getHostelRooms(hostelId),
        Api().getOwnerStudents(),
      ]);
      final data = results[0];
      final students = results[1];

      final assignedBedsByRoom = <String, Set<String>>{};
      for (final rawStudent in students) {
        if (rawStudent is! Map) continue;
        final student = Map<String, dynamic>.from(rawStudent);
        final studentHostelId = int.tryParse(
          student['hostel_id']?.toString() ?? '',
        );
        if (studentHostelId != hostelId) continue;
        final roomNumber = student['room_number']?.toString().trim();
        final bedNumber = student['bed_number']?.toString().trim();
        if (roomNumber == null ||
            roomNumber.isEmpty ||
            bedNumber == null ||
            bedNumber.isEmpty) {
          continue;
        }
        assignedBedsByRoom
            .putIfAbsent(roomNumber.toLowerCase(), () => <String>{})
            .add(bedNumber.toLowerCase());
      }

      final loadedRooms = data.map<Map<String, dynamic>>((item) {
        final room = Map<String, dynamic>.from(item as Map);

        final capacity = int.tryParse(room['capacity']?.toString() ?? '0') ?? 0;

        final available =
            int.tryParse(room['available_capacity']?.toString() ?? '0') ?? 0;

        final occupied = (capacity - available).clamp(0, capacity).toInt();
        final roomNumber = room['room_number']?.toString() ?? '';
        final assignedBeds =
            assignedBedsByRoom[roomNumber.toLowerCase()] ?? <String>{};
        final inventoryBeds =
            (room['beds'] as List?)
                ?.whereType<Map>()
                .map((bed) => Map<String, dynamic>.from(bed))
                .toList() ??
            const <Map<String, dynamic>>[];
        final bedInventory = inventoryBeds.isNotEmpty
            ? inventoryBeds
            : List<Map<String, dynamic>>.generate(
                capacity,
                (index) => {'bed_number': '${index + 1}'},
              );
        final beds = bedInventory.asMap().entries.map((entry) {
          final bed = entry.value;
          final bedNumber = bed['bed_number']?.toString() ?? '${entry.key + 1}';
          final normalized = bedNumber.toLowerCase().replaceAll(
            RegExp(r'[^a-z0-9]'),
            '',
          );
          final aliases = {normalized, 'bed$normalized', 'b$normalized'};
          final knownOccupied =
              bed['occupied'] == true ||
              bed['status']?.toString() == 'occupied' ||
              assignedBeds.any(
                (bed) =>
                    aliases.contains(bed.replaceAll(RegExp(r'[^a-z0-9]'), '')),
              );
          return {
            'id': bed['id'],
            'number': bedNumber,
            'occupied': knownOccupied,
          };
        }).toList();

        // Older bookings may not have a saved bed number. Keep their occupied
        // count visible by filling the first still-empty inventory beds.
        var unnumberedOccupied =
            occupied - beds.where((bed) => bed['occupied'] == true).length;
        for (final bed in beds) {
          if (unnumberedOccupied <= 0) break;
          if (bed['occupied'] == true) continue;
          bed['occupied'] = true;
          unnumberedOccupied--;
        }

        String status;

        if (available <= 0) {
          status = 'Full';
        } else {
          status = 'Available';
        }

        return {
          'id': room['id'],
          'hostelId': room['hostel_id'],
          'roomNo': roomNumber,
          'floor': room['floor_label']?.toString().trim().isNotEmpty == true
              ? room['floor_label'].toString()
              : 'Ground Floor',
          'type': room['room_type']?.toString() ?? 'Shared',
          'totalBeds': capacity,
          'occupiedBeds': beds.where((bed) => bed['occupied'] == true).length,
          'beds': beds,
          'rent': int.tryParse(room['monthly_rent']?.toString() ?? '0') ?? 0,
          'acIncluded': room['ac_included'] == true,
          'amenities': (room['amenities'] as List? ?? const [])
              .map((item) => item.toString())
              .toList(),
          'notes': room['notes']?.toString() ?? '',
          'status': status,
        };
      }).toList();

      if (!mounted) return;

      setState(() {
        _rooms
          ..clear()
          ..addAll(loadedRooms);

        _floors
          ..clear()
          ..add('All Floors')
          ..addAll(
            loadedRooms.map((room) => room['floor'].toString()).toSet().toList()
              ..sort(),
          );

        _loadingRooms = false;
      });
    } catch (e) {
      debugPrint('Owner rooms load failed: $e');

      if (!mounted) return;

      setState(() {
        _loadingRooms = false;
      });

      ScaffoldMessenger.of(context)
        ..hideCurrentSnackBar()
        ..showSnackBar(SnackBar(content: Text('Could not load rooms: $e')));
    }
  }

  @override
  void dispose() {
    _anim.dispose();
    _searchCtrl.dispose();
    super.dispose();
  }

  // ===========================================================================
  // FILTERED ROOMS
  // ===========================================================================

  List<Map<String, dynamic>> get _filteredRooms {
    final query = _searchCtrl.text.trim().toLowerCase();

    return _rooms.where((room) {
      final roomNo = room['roomNo'].toString().toLowerCase();

      final floor = room['floor'].toString();

      final status = room['status'].toString();

      final matchesSearch = query.isEmpty || roomNo.contains(query);

      final matchesStatus =
          _selectedFilter == 'All' || status == _selectedFilter;

      final matchesFloor =
          _selectedFloor == 'All Floors' || floor == _selectedFloor;

      return matchesSearch && matchesStatus && matchesFloor;
    }).toList();
  }

  int get _totalRooms => _rooms.length;

  int get _totalBeds {
    return _rooms.fold<int>(
      0,
      (sum, room) => sum + ((room['totalBeds'] as num?)?.toInt() ?? 0),
    );
  }

  int get _occupiedBeds {
    return _rooms.fold<int>(
      0,
      (sum, room) => sum + ((room['occupiedBeds'] as num?)?.toInt() ?? 0),
    );
  }

  int get _availableBeds =>
      (_totalBeds - _occupiedBeds).clamp(0, 999999).toInt();

  double get _occupancy {
    if (_totalBeds == 0) return 0;

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
        floatingActionButton: _buildFab(),

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
                    sliver: SliverToBoxAdapter(child: _buildSummary()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(16, 22, 16, 0),
                    sliver: SliverToBoxAdapter(child: _buildSearch()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(16, 14, 16, 0),
                    sliver: SliverToBoxAdapter(child: _buildStatusFilters()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(16, 14, 16, 0),
                    sliver: SliverToBoxAdapter(child: _buildFloorSelector()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(16, 24, 16, 10),
                    sliver: SliverToBoxAdapter(child: _buildSectionHeader()),
                  ),

                  _buildRoomsSliver(),

                  const SliverToBoxAdapter(child: SizedBox(height: 110)),
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
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              _glassCircleButton(
                icon: Icons.arrow_back_rounded,
                onTap: _goBackToDashboard,
              ),

              const Spacer(),

              _glassCircleButton(
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
                  const Color(0xFF073F42).withOpacity(0.94),
                  const Color(0xFF032E31).withOpacity(0.92),
                  Colors.black.withOpacity(0.60),
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
              children: [
                Container(
                  width: 56,
                  height: 56,
                  decoration: BoxDecoration(
                    color: AppTheme.teal.withOpacity(0.26),
                    borderRadius: BorderRadius.circular(17),
                    border: Border.all(
                      color: AppTheme.tealMint.withOpacity(0.20),
                    ),
                  ),
                  child: const Icon(
                    Icons.meeting_room_rounded,
                    color: AppTheme.tealMint,
                    size: 28,
                  ),
                ),

                const SizedBox(width: 13),

                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Rooms & Beds',
                        style: GoogleFonts.playfairDisplay(
                          color: Colors.white,
                          fontSize: 27,
                          height: 1,
                          fontWeight: FontWeight.w700,
                        ),
                      ),

                      const SizedBox(height: 7),

                      Text(
                        widget.hostelName,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: GoogleFonts.dmSans(
                          color: Colors.white.withOpacity(0.58),
                          fontSize: 10.5,
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
                    '$_totalRooms rooms',
                    style: GoogleFonts.dmSans(
                      color: AppTheme.tealMint,
                      fontSize: 9.5,
                      fontWeight: FontWeight.w700,
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

  Widget _glassCircleButton({
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
            color: Colors.black.withOpacity(0.28),
            shape: BoxShape.circle,
            border: Border.all(color: Colors.white.withOpacity(0.16)),
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

  // ===========================================================================
  // SUMMARY
  // ===========================================================================

  Widget _buildSummary() {
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
                      'Room Overview',
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 13,
                        fontWeight: FontWeight.w800,
                      ),
                    ),

                    const SizedBox(height: 3),

                    Text(
                      'Current accommodation status',
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.46),
                        fontSize: 8.7,
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
                  color: AppTheme.teal.withOpacity(0.16),
                  borderRadius: BorderRadius.circular(18),
                ),
                child: Text(
                  '${(_occupancy * 100).round()}% occupied',
                  style: GoogleFonts.dmSans(
                    color: AppTheme.tealMint,
                    fontSize: 9.5,
                    fontWeight: FontWeight.w800,
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

          const SizedBox(height: 17),

          Row(
            children: [
              Expanded(
                child: _summaryItem(
                  value: '$_totalRooms',
                  label: 'Rooms',
                  icon: Icons.meeting_room_rounded,
                ),
              ),
              _summaryDivider(),
              Expanded(
                child: _summaryItem(
                  value: '$_totalBeds',
                  label: 'Beds',
                  icon: Icons.bed_rounded,
                ),
              ),
              _summaryDivider(),
              Expanded(
                child: _summaryItem(
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

  Widget _summaryItem({
    required String value,
    required String label,
    required IconData icon,
  }) {
    return Column(
      children: [
        Container(
          width: 38,
          height: 38,
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.17),
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, size: 18, color: AppTheme.tealMint),
        ),

        const SizedBox(height: 7),

        Text(
          value,
          style: GoogleFonts.dmSans(
            color: Colors.white,
            fontSize: 17,
            fontWeight: FontWeight.w800,
          ),
        ),

        const SizedBox(height: 2),

        Text(
          label,
          textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(
            color: Colors.white.withOpacity(0.43),
            fontSize: 8.5,
          ),
        ),
      ],
    );
  }

  Widget _summaryDivider() {
    return Container(
      width: 1,
      height: 48,
      color: Colors.white.withOpacity(0.08),
    );
  }

  // ===========================================================================
  // SEARCH
  // ===========================================================================

  Widget _buildSearch() {
    return AnimatedContainer(
      duration: const Duration(milliseconds: 180),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.teal.withOpacity(0.22)),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.12),
            blurRadius: 14,
            offset: const Offset(0, 5),
          ),
        ],
      ),

      child: TextField(
        controller: _searchCtrl,
        onChanged: (_) {
          setState(() {});
        },
        style: GoogleFonts.dmSans(
          fontSize: 13,
          fontWeight: FontWeight.w500,
          color: const Color(0xFF143637),
        ),
        decoration: InputDecoration(
          hintText: 'Search room number...',
          hintStyle: GoogleFonts.dmSans(
            color: const Color(0xFF8D9999),
            fontSize: 12.5,
          ),
          prefixIcon: const Icon(
            Icons.search_rounded,
            color: Color(0xFF0A6668),
            size: 20,
          ),
          suffixIcon: _searchCtrl.text.isNotEmpty
              ? IconButton(
                  onPressed: () {
                    _searchCtrl.clear();

                    setState(() {});
                  },
                  icon: const Icon(
                    Icons.close_rounded,
                    size: 18,
                    color: Color(0xFF0A6668),
                  ),
                )
              : null,
          border: InputBorder.none,
          contentPadding: const EdgeInsets.symmetric(vertical: 15),
        ),
      ),
    );
  }

  // ===========================================================================
  // STATUS FILTERS
  // ===========================================================================

  Widget _buildStatusFilters() {
    return SizedBox(
      height: 40,
      child: ListView.separated(
        physics: const BouncingScrollPhysics(),
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
              padding: const EdgeInsets.symmetric(horizontal: 15, vertical: 8),
              decoration: BoxDecoration(
                color: selected ? AppTheme.teal : const Color(0xC909292B),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(
                  color: selected
                      ? AppTheme.tealMint.withOpacity(0.34)
                      : Colors.white.withOpacity(0.10),
                ),
              ),
              child: Text(
                filter,
                style: GoogleFonts.dmSans(
                  color: selected
                      ? Colors.white
                      : Colors.white.withOpacity(0.62),
                  fontSize: 10.5,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  // ===========================================================================
  // FLOOR SELECTOR
  // ===========================================================================

  Widget _buildFloorSelector() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 13),
      decoration: BoxDecoration(
        color: const Color(0xD909292B),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white.withOpacity(0.10)),
      ),
      child: Row(
        children: [
          const Icon(Icons.layers_outlined, size: 18, color: AppTheme.tealMint),

          const SizedBox(width: 10),

          Expanded(
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                value: _selectedFloor,
                isExpanded: true,
                dropdownColor: const Color(0xFF082C2F),
                iconEnabledColor: AppTheme.tealMint,
                borderRadius: BorderRadius.circular(16),
                items: _floors.map((floor) {
                  return DropdownMenuItem<String>(
                    value: floor,
                    child: Text(
                      floor,
                      style: GoogleFonts.dmSans(
                        fontSize: 11,
                        color: Colors.white,
                      ),
                    ),
                  );
                }).toList(),
                onChanged: (value) {
                  if (value == null) {
                    return;
                  }

                  setState(() {
                    _selectedFloor = value;
                  });
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // SECTION HEADER
  // ===========================================================================

  Widget _buildSectionHeader() {
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
                'Rooms',
                style: GoogleFonts.dmSans(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.w800,
                ),
              ),

              const SizedBox(height: 2),

              Text(
                'Manage room availability and beds',
                style: GoogleFonts.dmSans(
                  color: Colors.white.withOpacity(0.54),
                  fontSize: 9,
                ),
              ),
            ],
          ),
        ),

        Container(
          padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.16),
            borderRadius: BorderRadius.circular(16),
          ),
          child: Text(
            '${_filteredRooms.length} found',
            style: GoogleFonts.dmSans(
              color: AppTheme.tealMint,
              fontSize: 9,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ],
    );
  }

  // ===========================================================================
  // ROOMS
  // ===========================================================================

  Widget _buildRoomsSliver() {
    final rooms = _filteredRooms;
    final visibleFloors = _selectedFloor == 'All Floors'
        ? _floors.where((floor) => floor != 'All Floors').toList()
        : <String>[_selectedFloor];

    if (visibleFloors.isEmpty) {
      return SliverPadding(
        padding: const EdgeInsets.fromLTRB(16, 18, 16, 0),
        sliver: SliverToBoxAdapter(child: _buildEmptyState()),
      );
    }

    final groupedRooms = <String, List<Map<String, dynamic>>>{};
    for (final room in rooms) {
      groupedRooms.putIfAbsent(room['floor'].toString(), () => []).add(room);
    }

    return SliverPadding(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      sliver: SliverToBoxAdapter(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            for (final floor in visibleFloors) ...[
              _floorHeader(floor, groupedRooms[floor]?.length ?? 0),
              const SizedBox(height: 10),
              if (groupedRooms[floor]?.isNotEmpty == true)
                for (final room in groupedRooms[floor]!)
                  Padding(
                    padding: const EdgeInsets.only(bottom: 13),
                    child: _buildRoomCard(room),
                  )
              else
                _emptyFloorCard(floor),
            ],
          ],
        ),
      ),
    );
  }

  Widget _floorHeader(String floor, int roomCount) {
    return Row(
      children: [
        Container(
          width: 4,
          height: 31,
          decoration: BoxDecoration(
            color: AppTheme.tealMint,
            borderRadius: BorderRadius.circular(8),
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                floor,
                style: GoogleFonts.dmSans(
                  color: Colors.white,
                  fontSize: 13,
                  fontWeight: FontWeight.w800,
                ),
              ),
              Text(
                '$roomCount room${roomCount == 1 ? '' : 's'} · tap a room for details',
                style: GoogleFonts.dmSans(
                  color: Colors.white.withOpacity(0.52),
                  fontSize: 8.5,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _emptyFloorCard(String floor) {
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xA6083032),
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: Colors.white.withValues(alpha: 0.09)),
      ),
      child: Row(
        children: [
          const Icon(Icons.meeting_room_outlined, color: AppTheme.tealMint),
          const SizedBox(width: 11),
          Expanded(
            child: Text(
              'No rooms on $floor yet',
              style: GoogleFonts.dmSans(
                color: Colors.white.withValues(alpha: 0.72),
                fontSize: 10,
              ),
            ),
          ),
          TextButton.icon(
            onPressed: _openAddRoom,
            icon: const Icon(Icons.add_rounded, size: 16),
            label: const Text('Add room'),
            style: TextButton.styleFrom(foregroundColor: AppTheme.tealMint),
          ),
        ],
      ),
    );
  }

  Widget _buildRoomCard(Map<String, dynamic> room) {
    final totalBeds = (room['totalBeds'] as num?)?.toInt() ?? 0;

    final occupiedBeds = (room['occupiedBeds'] as num?)?.toInt() ?? 0;

    final availableBeds = (totalBeds - occupiedBeds)
        .clamp(0, totalBeds)
        .toInt();

    final status = room['status']?.toString() ?? 'Available';

    final statusColor = _statusColor(status);

    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () {
          _openRoomDetails(room);
        },
        borderRadius: BorderRadius.circular(20),

        child: Container(
          padding: const EdgeInsets.all(15),

          decoration: BoxDecoration(
            color: const Color(0xD909292B),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: Colors.white.withOpacity(0.09)),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.10),
                blurRadius: 15,
                offset: const Offset(0, 6),
              ),
            ],
          ),

          child: Column(
            children: [
              Row(
                children: [
                  Container(
                    width: 48,
                    height: 48,
                    decoration: BoxDecoration(
                      color: AppTheme.teal.withOpacity(0.18),
                      borderRadius: BorderRadius.circular(14),
                    ),
                    child: const Icon(
                      Icons.meeting_room_rounded,
                      color: AppTheme.tealMint,
                      size: 23,
                    ),
                  ),

                  const SizedBox(width: 12),

                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: Text(
                                'Room ${room['roomNo']}',
                                maxLines: 1,
                                overflow: TextOverflow.ellipsis,
                                style: GoogleFonts.dmSans(
                                  color: Colors.white,
                                  fontSize: 13,
                                  fontWeight: FontWeight.w800,
                                ),
                              ),
                            ),

                            const SizedBox(width: 7),

                            _statusChip(status, statusColor),
                          ],
                        ),

                        const SizedBox(height: 5),

                        Text(
                          '${room['floor']} · ${room['type']}',
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.dmSans(
                            color: Colors.white.withOpacity(0.49),
                            fontSize: 9,
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(width: 5),

                  const Icon(
                    Icons.chevron_right_rounded,
                    color: AppTheme.tealMint,
                    size: 19,
                  ),
                ],
              ),

              const SizedBox(height: 14),

              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(
                  vertical: 12,
                  horizontal: 5,
                ),
                decoration: BoxDecoration(
                  color: Colors.black.withOpacity(0.16),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.white.withOpacity(0.06)),
                ),

                child: Row(
                  children: [
                    Expanded(
                      child: _roomMiniStat(
                        label: 'Occupied',
                        value: '$occupiedBeds',
                        icon: Icons.person_rounded,
                      ),
                    ),

                    _miniDivider(),

                    Expanded(
                      child: _roomMiniStat(
                        label: 'Available',
                        value: '$availableBeds',
                        icon: Icons.bed_rounded,
                      ),
                    ),

                    _miniDivider(),

                    Expanded(
                      child: _roomMiniStat(
                        label: 'Rent',
                        value:
                            'PKR ${_formatAmount((room['rent'] as num?)?.toInt() ?? 0)}',
                        icon: Icons.payments_outlined,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 12),

              _bedStatusGrid(room),
            ],
          ),
        ),
      ),
    );
  }

  Widget _bedStatusGrid(Map<String, dynamic> room) {
    final beds =
        (room['beds'] as List?)
            ?.whereType<Map>()
            .map((bed) => Map<String, dynamic>.from(bed))
            .toList() ??
        const <Map<String, dynamic>>[];

    if (beds.isEmpty) return const SizedBox.shrink();

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(11),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.12),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white.withOpacity(0.07)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                'BED STATUS',
                style: GoogleFonts.dmSans(
                  color: Colors.white.withOpacity(0.52),
                  fontSize: 8,
                  fontWeight: FontWeight.w800,
                  letterSpacing: 0.7,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 7,
            runSpacing: 7,
            children: beds.map((bed) {
              final occupied = bed['occupied'] == true;
              final color = occupied
                  ? const Color(0xFFFFC978)
                  : AppTheme.tealMint;
              return Container(
                padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 7),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.13),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: color.withOpacity(0.32)),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      occupied ? Icons.person_rounded : Icons.bed_rounded,
                      size: 13,
                      color: color,
                    ),
                    const SizedBox(width: 5),
                    Text(
                      'Bed ${bed['number']}',
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 9,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                    const SizedBox(width: 4),
                    Text(
                      occupied ? 'Occupied' : 'Empty',
                      style: GoogleFonts.dmSans(
                        color: color,
                        fontSize: 8,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ),
        ],
      ),
    );
  }

  Widget _roomMiniStat({
    required String label,
    required String value,
    required IconData icon,
  }) {
    return Column(
      children: [
        Icon(icon, size: 15, color: AppTheme.tealMint),

        const SizedBox(height: 5),

        Text(
          value,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          textAlign: TextAlign.center,
          style: GoogleFonts.dmSans(
            color: Colors.white,
            fontSize: 9.5,
            fontWeight: FontWeight.w700,
          ),
        ),

        const SizedBox(height: 2),

        Text(
          label,
          maxLines: 1,
          overflow: TextOverflow.ellipsis,
          style: GoogleFonts.dmSans(
            color: Colors.white.withOpacity(0.42),
            fontSize: 7.6,
          ),
        ),
      ],
    );
  }

  Widget _miniDivider() {
    return Container(
      width: 1,
      height: 34,
      color: Colors.white.withOpacity(0.07),
    );
  }

  Widget _statusChip(String status, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(0.28)),
      ),
      child: Text(
        status,
        style: GoogleFonts.dmSans(
          color: color,
          fontSize: 8,
          fontWeight: FontWeight.w800,
        ),
      ),
    );
  }

  Color _statusColor(String status) {
    switch (status) {
      case 'Available':
        return const Color(0xFF66D8A6);

      case 'Full':
        return const Color(0xFFFFB05F);

      case 'Maintenance':
        return const Color(0xFFFF7B7B);

      default:
        return AppTheme.tealMint;
    }
  }

  // ===========================================================================
  // ROOM DETAILS
  // ===========================================================================

  Future<void> _openRoomDetails(Map<String, dynamic> room) async {
    final result = await Navigator.push<Map<String, dynamic>>(
      context,
      MaterialPageRoute(
        builder: (_) =>
            RoomDetailsScreen(hostelName: widget.hostelName, room: room),
      ),
    );

    if (!mounted || result == null) {
      return;
    }

    if (result['deleted'] == true) {
      final roomNo = result['roomNo']?.toString();

      setState(() {
        _rooms.removeWhere((item) => item['roomNo']?.toString() == roomNo);
      });

      _showSuccess('Room $roomNo deleted');
    }
  }

  // ===========================================================================
  // EMPTY STATE
  // ===========================================================================

  Widget _buildEmptyState() {
    return _glassCard(
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 28, horizontal: 14),
        child: Column(
          children: [
            Container(
              width: 68,
              height: 68,
              decoration: BoxDecoration(
                color: AppTheme.teal.withOpacity(0.16),
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.bed_outlined,
                size: 30,
                color: AppTheme.tealMint,
              ),
            ),

            const SizedBox(height: 15),

            Text(
              'No rooms found',
              style: GoogleFonts.playfairDisplay(
                color: Colors.white,
                fontSize: 20,
                fontWeight: FontWeight.w700,
              ),
            ),

            const SizedBox(height: 6),

            Text(
              'Try changing the filters or add a new room.',
              textAlign: TextAlign.center,
              style: GoogleFonts.dmSans(
                color: Colors.white.withOpacity(0.50),
                fontSize: 10,
                height: 1.45,
              ),
            ),
          ],
        ),
      ),
    );
  }

  // ===========================================================================
  // ADD ROOM
  // ===========================================================================

  Widget _buildFab() {
    return FloatingActionButton.extended(
      onPressed: _openAddRoom,
      elevation: 4,
      backgroundColor: AppTheme.teal,
      foregroundColor: Colors.white,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
      icon: const Icon(Icons.add_rounded, size: 20),
      label: Text(
        'Add Room',
        style: GoogleFonts.dmSans(fontWeight: FontWeight.w800, fontSize: 11),
      ),
    );
  }

  Future<void> _openAddRoom() async {
    final result = await Navigator.push<Map<String, dynamic>>(
      context,
      MaterialPageRoute(
        builder: (_) => AddRoomScreen(
          hostelName: widget.hostelName,
          availableFloors: _floors
              .where((floor) => floor != 'All Floors')
              .toList(),
          initialFloor: _selectedFloor == 'All Floors' ? null : _selectedFloor,
        ),
      ),
    );

    if (!mounted || result == null) {
      return;
    }

    final newRoom = Map<String, dynamic>.from(result);

    newRoom['totalBeds'] = (newRoom['totalBeds'] as num?)?.toInt() ?? 0;

    newRoom['occupiedBeds'] = (newRoom['occupiedBeds'] as num?)?.toInt() ?? 0;

    newRoom['rent'] = (newRoom['rent'] as num?)?.toInt() ?? 0;

    newRoom['beds'] = List.generate(
      newRoom['totalBeds'] as int,
      (index) => {'number': '${index + 1}', 'occupied': false},
    );

    setState(() {
      _rooms.insert(0, newRoom);
      final floor = newRoom['floor']?.toString() ?? 'Ground Floor';
      if (!_floors.contains(floor)) {
        _floors.add(floor);
      }
    });

    _showSuccess('Room ${newRoom['roomNo']} added successfully');
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
              border: Border.all(color: AppTheme.tealMint.withOpacity(0.18)),
            ),

            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Center(
                  child: Container(
                    width: 40,
                    height: 4,
                    margin: const EdgeInsets.only(bottom: 12),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.18),
                      borderRadius: BorderRadius.circular(3),
                    ),
                  ),
                ),

                _menuTile(
                  icon: Icons.picture_as_pdf_outlined,
                  title: 'Export rooms PDF',
                  subtitle: 'Download floors, rooms, beds and amenities',
                  onTap: () async {
                    Navigator.pop(sheetContext);
                    await _exportRoomPdf();
                  },
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _exportRoomPdf() async {
    if (_rooms.isEmpty) {
      _showTemporary('Add at least one room before exporting a report.');
      return;
    }
    try {
      final document = pw.Document();
      final grouped = <String, List<Map<String, dynamic>>>{};
      for (final room in _rooms) {
        grouped.putIfAbsent(room['floor'].toString(), () => []).add(room);
      }
      document.addPage(
        pw.MultiPage(
          pageFormat: PdfPageFormat.a4,
          margin: const pw.EdgeInsets.all(28),
          build: (context) => [
            pw.Text(
              'StayBuddy Room Inventory',
              style: pw.TextStyle(fontSize: 21, fontWeight: pw.FontWeight.bold),
            ),
            pw.SizedBox(height: 4),
            pw.Text(widget.hostelName),
            pw.SizedBox(height: 14),
            ...grouped.entries.expand(
              (entry) => [
                pw.Header(level: 1, text: entry.key),
                pw.Table.fromTextArray(
                  headerStyle: pw.TextStyle(
                    fontWeight: pw.FontWeight.bold,
                    color: PdfColors.white,
                  ),
                  headerDecoration: const pw.BoxDecoration(
                    color: PdfColor.fromInt(0xFF087C80),
                  ),
                  cellStyle: const pw.TextStyle(fontSize: 8),
                  headers: const [
                    'Room',
                    'Seater',
                    'Rent',
                    'Beds',
                    'Availability',
                    'Amenities',
                  ],
                  data: entry.value.map((room) {
                    final beds = (room['beds'] as List? ?? const [])
                        .map((bed) => Map<String, dynamic>.from(bed as Map))
                        .toList();
                    final available = beds
                        .where((bed) => bed['occupied'] != true)
                        .length;
                    final labels = beds
                        .map((bed) => bed['number']?.toString() ?? '-')
                        .join(', ');
                    final amenities = (room['amenities'] as List? ?? const [])
                        .join(', ');
                    return [
                      room['roomNo'].toString(),
                      '${room['totalBeds']} Seater',
                      'PKR ${room['rent']}',
                      labels,
                      '$available / ${room['totalBeds']} empty',
                      amenities.isEmpty ? 'None' : amenities,
                    ];
                  }).toList(),
                ),
                pw.SizedBox(height: 12),
              ],
            ),
          ],
        ),
      );
      await Printing.sharePdf(
        bytes: await document.save(),
        filename:
            '${widget.hostelName.replaceAll(RegExp(r'[^A-Za-z0-9]'), '_')}_rooms.pdf',
      );
    } catch (error) {
      if (mounted) _showTemporary('Could not create PDF: $error');
    }
  }

  Widget _menuTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 7),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(15),
        child: Container(
          padding: const EdgeInsets.all(11),
          decoration: BoxDecoration(
            color: Colors.black.withOpacity(0.16),
            borderRadius: BorderRadius.circular(15),
            border: Border.all(color: Colors.white.withOpacity(0.07)),
          ),

          child: Row(
            children: [
              Container(
                width: 41,
                height: 41,
                decoration: BoxDecoration(
                  color: AppTheme.teal.withOpacity(0.18),
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Icon(icon, color: AppTheme.tealMint, size: 19),
              ),

              const SizedBox(width: 11),

              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 11.5,
                        fontWeight: FontWeight.w700,
                      ),
                    ),

                    const SizedBox(height: 2),

                    Text(
                      subtitle,
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.45),
                        fontSize: 8.5,
                      ),
                    ),
                  ],
                ),
              ),

              const Icon(
                Icons.chevron_right_rounded,
                color: AppTheme.tealMint,
                size: 18,
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ===========================================================================
  // GLASS CARD
  // ===========================================================================

  Widget _glassCard({
    required Widget child,
    EdgeInsetsGeometry padding = const EdgeInsets.all(15),
  }) {
    return Container(
      width: double.infinity,
      padding: padding,
      decoration: BoxDecoration(
        color: const Color(0xD909292B),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white.withOpacity(0.10)),
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

  // ===========================================================================
  // MESSAGES
  // ===========================================================================

  void _showSuccess(String text) {
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
                  text,
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
      );
  }

  void _showTemporary(String text) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          margin: const EdgeInsets.all(16),
          backgroundColor: const Color(0xFF123F40),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(14),
          ),
          content: Text(
            text,
            style: GoogleFonts.dmSans(color: Colors.white, fontSize: 10.5),
          ),
        ),
      );
  }

  // ===========================================================================
  // FORMAT AMOUNT
  // ===========================================================================

  String _formatAmount(int amount) {
    final text = amount.toString();

    final buffer = StringBuffer();

    for (int i = 0; i < text.length; i++) {
      final remaining = text.length - i;

      buffer.write(text[i]);

      if (remaining > 1 && remaining % 3 == 1) {
        buffer.write(',');
      }
    }

    return buffer.toString();
  }
}
