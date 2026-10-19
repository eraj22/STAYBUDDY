import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../../../theme.dart';
import '../../../widgets/responsive_shell.dart';
import '../../../widgets/video_background.dart';
import '../../../api.dart';
import 'hostel_profile_editor_screen.dart';

class OwnerHostelScreen extends StatefulWidget {
  final String ownerName;
  final String ownerEmail;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;
  final VoidCallback? onBackToDashboard;

  const OwnerHostelScreen({
    super.key,
    required this.ownerName,
    required this.ownerEmail,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
    this.onBackToDashboard,
  });

  @override
  State<OwnerHostelScreen> createState() => _OwnerHostelScreenState();
}

class _OwnerHostelScreenState extends State<OwnerHostelScreen>
    with SingleTickerProviderStateMixin {
  late String _hostelName;
  late String _hostelType;
  late String _city;
  late String _address;

  String _phone = '0300-1234567';
  String _email = 'staybuddyhostel@gmail.com';

  int _totalRooms = 24;
  int _totalBeds = 62;
  int _occupiedBeds = 44;

  bool _isActive = true;
  bool _verified = true;
  int? _hostelId;

  late final AnimationController _anim;
  late final Animation<double> _fade;
  late final Animation<double> _slide;

  List<Map<String, dynamic>> _amenities = [
    {
      'name': 'WiFi',
      'icon': Icons.wifi_rounded,
      'color': const Color(0xFF4FA7B5),
    },
    {
      'name': 'CCTV',
      'icon': Icons.videocam_rounded,
      'color': const Color(0xFF7C9BA1),
    },
    {
      'name': 'Laundry',
      'icon': Icons.local_laundry_service_rounded,
      'color': const Color(0xFFA783D6),
    },
    {
      'name': 'Mess',
      'icon': Icons.restaurant_rounded,
      'color': const Color(0xFFE3A452),
    },
    {
      'name': 'Parking',
      'icon': Icons.local_parking_rounded,
      'color': const Color(0xFF67B78C),
    },
    {
      'name': 'Security',
      'icon': Icons.security_rounded,
      'color': AppTheme.tealMint,
    },
  ];

  List<String> _rules = [
    'Gate closes at 10:00 PM',
    'Visitors must register at reception',
    'No smoking inside hostel premises',
    'Maintain cleanliness in shared areas',
    'Keep noise low after 11:00 PM',
  ];
  @override
  void initState() {
    super.initState();

    _hostelName = widget.hostelName;
    _hostelType = widget.hostelType;
    _city = widget.city;
    _address = widget.address;

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
    _loadHostelData();
  }

  Future<void> _loadHostelData() async {
    try {
      final results = await Future.wait([
        Api().getOwnerHostels(),
        Api().getOwnerDashboard(),
      ]);
      final hostels = results[0] as List<dynamic>;
      final dashboard = results[1] as Map<String, dynamic>;
      if (hostels.isEmpty) {
        return;
      }

      Map<String, dynamic>? hostel;

      for (final item in hostels) {
        if (item is Map) {
          final data = Map<String, dynamic>.from(item);
          if (data['name']?.toString().trim().toLowerCase() ==
              widget.hostelName.trim().toLowerCase()) {
            hostel = data;
            break;
          }
        }
      }

      hostel ??= Map<String, dynamic>.from(hostels.first as Map);

      Map<String, dynamic>? summary;
      final dashboardHostels = dashboard['hostels'];
      if (dashboardHostels is List) {
        for (final item in dashboardHostels.whereType<Map>()) {
          final data = Map<String, dynamic>.from(item);
          if (data['hostel_id']?.toString() == hostel['id']?.toString()) {
            summary = data;
            break;
          }
        }
      }

      final amenities = hostel['amenities'] is List
          ? List<dynamic>.from(hostel['amenities'])
          : <dynamic>[];
      final rules = hostel['rules'] is List
          ? List<dynamic>.from(hostel['rules'])
          : <dynamic>[];

      if (!mounted) return;

      setState(() {
        _hostelId = int.tryParse(hostel!['id']?.toString() ?? '');
        _hostelName = hostel['name']?.toString() ?? widget.hostelName;
        _hostelType = hostel['hostel_type']?.toString() ?? widget.hostelType;
        _city = hostel['city']?.toString() ?? widget.city;
        _address = hostel['address']?.toString() ?? widget.address;
        _phone = hostel['contact_phone']?.toString() ?? '';
        _email = hostel['contact_email']?.toString() ?? '';
        _verified = hostel['verified'] == true;
        // A saved owner hostel stays operational for bookings. Publication is
        // only a discovery/listing concern, not an "inactive hostel" state.
        final listingState =
            hostel['publication_status']?.toString().toLowerCase() ?? '';
        _isActive = !['inactive', 'archived', 'deactivated']
            .contains(listingState);
        _totalRooms =
            int.tryParse(
              (summary?['total_rooms'] ?? hostel['total_rooms'])?.toString() ??
                  '0',
            ) ??
            0;
        _totalBeds =
            int.tryParse(
              (summary?['total'] ?? hostel['total_capacity'])?.toString() ??
                  '0',
            ) ??
            0;
        _occupiedBeds =
            int.tryParse(summary?['occupied']?.toString() ?? '0') ?? 0;
        if (amenities.isNotEmpty) {
          _amenities = amenities
              .map((name) => _amenityFor(name.toString()))
              .toList();
        }
        if (rules.isNotEmpty) {
          _rules = rules.map((rule) => rule.toString()).toList();
        }
      });
    } catch (e) {
      debugPrint('Owner hostel load failed: $e');
    }
  }

  Map<String, dynamic> _amenityFor(String name) {
    final lower = name.toLowerCase();
    IconData icon = Icons.check_circle_outline_rounded;
    if (lower.contains('wifi')) icon = Icons.wifi_rounded;
    if (lower.contains('cctv')) icon = Icons.videocam_rounded;
    if (lower.contains('laundry')) icon = Icons.local_laundry_service_rounded;
    if (lower.contains('mess')) icon = Icons.restaurant_rounded;
    if (lower.contains('parking')) icon = Icons.local_parking_rounded;
    if (lower.contains('security')) icon = Icons.security_rounded;
    return {'name': name, 'icon': icon, 'color': AppTheme.tealMint};
  }

  @override
  void dispose() {
    _anim.dispose();
    super.dispose();
  }

  int get _availableBeds => (_totalBeds - _occupiedBeds).clamp(0, _totalBeds);

  double get _occupancy {
    if (_totalBeds <= 0) return 0;

    return (_occupiedBeds / _totalBeds).clamp(0.0, 1.0).toDouble();
  }

  int get _occupancyPercent => (_occupancy * 100).round();

  String get _typeText {
    if (_hostelType.trim().isEmpty) {
      return 'Hostel';
    }

    final value = _hostelType.trim().toLowerCase();

    return '${value[0].toUpperCase()}${value.substring(1)} Hostel';
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
                        title: 'Hostel Information',
                        subtitle: 'General details and location',
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
                        title: 'Capacity',
                        subtitle: 'Bed availability and occupancy',
                      ),
                    ),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    sliver: SliverToBoxAdapter(child: _buildCapacityCard()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(16, 24, 16, 10),
                    sliver: SliverToBoxAdapter(
                      child: _sectionTitle(
                        title: 'Amenities',
                        subtitle: 'Facilities available to residents',
                      ),
                    ),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    sliver: SliverToBoxAdapter(child: _buildAmenities()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(16, 24, 16, 10),
                    sliver: SliverToBoxAdapter(
                      child: _sectionTitle(
                        title: 'Hostel Rules',
                        subtitle: 'Resident policies and guidelines',
                      ),
                    ),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    sliver: SliverToBoxAdapter(child: _buildRulesCard()),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.fromLTRB(16, 24, 16, 10),
                    sliver: SliverToBoxAdapter(
                      child: _sectionTitle(
                        title: 'Contact & Ownership',
                        subtitle: 'Owner and hostel contact details',
                      ),
                    ),
                  ),

                  SliverPadding(
                    padding: const EdgeInsets.symmetric(horizontal: 16),
                    sliver: SliverToBoxAdapter(child: _buildContactCard()),
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
                icon: Icons.edit_rounded,
                onTap: _openEditHostel,
              ),

              const SizedBox(width: 8),

              _glassCircleButton(
                icon: Icons.more_horiz_rounded,
                onTap: _showMoreMenu,
              ),
            ],
          ),

          const SizedBox(height: 26),

          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(18),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
                colors: [
                  const Color(0xFF073F42).withOpacity(0.92),
                  const Color(0xFF032E31).withOpacity(0.92),
                  Colors.black.withOpacity(0.58),
                ],
              ),
              borderRadius: BorderRadius.circular(28),
              border: Border.all(color: AppTheme.tealMint.withOpacity(0.24)),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.22),
                  blurRadius: 24,
                  offset: const Offset(0, 10),
                ),
              ],
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      width: 60,
                      height: 60,
                      decoration: BoxDecoration(
                        color: AppTheme.teal.withOpacity(0.30),
                        borderRadius: BorderRadius.circular(18),
                        border: Border.all(
                          color: AppTheme.tealMint.withOpacity(0.24),
                        ),
                      ),
                      child: const Icon(
                        Icons.apartment_rounded,
                        color: AppTheme.tealMint,
                        size: 29,
                      ),
                    ),

                    const SizedBox(width: 13),

                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _hostelName,
                            maxLines: 2,
                            overflow: TextOverflow.ellipsis,
                            style: GoogleFonts.playfairDisplay(
                              color: Colors.white,
                              fontSize: 26,
                              height: 1.08,
                              fontWeight: FontWeight.w700,
                            ),
                          ),

                          const SizedBox(height: 7),

                          Row(
                            children: [
                              Icon(
                                Icons.location_on_outlined,
                                size: 14,
                                color: Colors.white.withOpacity(0.58),
                              ),

                              const SizedBox(width: 4),

                              Expanded(
                                child: Text(
                                  '${_city.isEmpty ? 'City not added' : _city} · $_typeText',
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: GoogleFonts.dmSans(
                                    color: Colors.white.withOpacity(0.62),
                                    fontSize: 10.5,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 17),

                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: [
                    if (_verified)
                      _headerChip(
                        icon: Icons.verified_rounded,
                        text: 'Verified',
                        color: AppTheme.tealMint,
                      ),

                    _headerChip(
                      icon: _isActive
                          ? Icons.check_circle_rounded
                          : Icons.pause_circle_filled_rounded,
                      text: _isActive ? 'Hostel Active' : 'Hostel Inactive',
                      color: _isActive
                          ? const Color(0xFF66D8A6)
                          : const Color(0xFFFF8A80),
                    ),
                  ],
                ),

                const SizedBox(height: 15),

                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.black.withOpacity(0.18),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: Colors.white.withOpacity(0.08)),
                  ),
                  child: Row(
                    children: [
                      const Icon(
                        Icons.location_on_rounded,
                        color: AppTheme.tealMint,
                        size: 18,
                      ),

                      const SizedBox(width: 9),

                      Expanded(
                        child: Text(
                          _address.isEmpty
                              ? 'Hostel address not added'
                              : _address,
                          maxLines: 2,
                          overflow: TextOverflow.ellipsis,
                          style: GoogleFonts.dmSans(
                            color: Colors.white.withOpacity(0.72),
                            fontSize: 9.5,
                            height: 1.4,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ],
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

  Widget _headerChip({
    required IconData icon,
    required String text,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.13),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.22)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: color, size: 13),
          const SizedBox(width: 5),
          Text(
            text,
            style: GoogleFonts.dmSans(
              color: color,
              fontSize: 9,
              fontWeight: FontWeight.w700,
            ),
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // OVERVIEW
  // ===========================================================================

  Widget _buildOverviewCard() {
    return _glassCard(
      child: LayoutBuilder(
        builder: (context, constraints) {
          if (constraints.maxWidth < 330) {
            return Column(
              children: [
                Row(
                  children: [
                    Expanded(
                      child: _overviewItem(
                        value: '$_totalRooms',
                        label: 'Rooms',
                        icon: Icons.meeting_room_rounded,
                      ),
                    ),
                    Expanded(
                      child: _overviewItem(
                        value: '$_totalBeds',
                        label: 'Beds',
                        icon: Icons.bed_rounded,
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 16),

                Row(
                  children: [
                    Expanded(
                      child: _overviewItem(
                        value: '$_occupiedBeds',
                        label: 'Occupied',
                        icon: Icons.groups_rounded,
                      ),
                    ),
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
            );
          }

          return Row(
            children: [
              Expanded(
                child: _overviewItem(
                  value: '$_totalRooms',
                  label: 'Rooms',
                  icon: Icons.meeting_room_rounded,
                ),
              ),

              _glassDivider(),

              Expanded(
                child: _overviewItem(
                  value: '$_totalBeds',
                  label: 'Beds',
                  icon: Icons.bed_rounded,
                ),
              ),

              _glassDivider(),

              Expanded(
                child: _overviewItem(
                  value: '$_occupiedBeds',
                  label: 'Occupied',
                  icon: Icons.groups_rounded,
                ),
              ),

              _glassDivider(),

              Expanded(
                child: _overviewItem(
                  value: '$_availableBeds',
                  label: 'Available',
                  icon: Icons.event_available_rounded,
                ),
              ),
            ],
          );
        },
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
          width: 39,
          height: 39,
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.18),
            borderRadius: BorderRadius.circular(13),
          ),
          child: Icon(icon, size: 19, color: AppTheme.tealMint),
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
            color: Colors.white.withOpacity(0.48),
            fontSize: 8.5,
          ),
        ),
      ],
    );
  }

  Widget _glassDivider() {
    return Container(
      width: 1,
      height: 48,
      margin: const EdgeInsets.symmetric(horizontal: 2),
      color: Colors.white.withOpacity(0.08),
    );
  }

  // ===========================================================================
  // SECTION TITLE
  // ===========================================================================

  Widget _sectionTitle({required String title, required String subtitle}) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
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
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.w800,
                ),
              ),

              const SizedBox(height: 2),

              Text(
                subtitle,
                style: GoogleFonts.dmSans(
                  color: Colors.white.withOpacity(0.56),
                  fontSize: 9.2,
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
            icon: Icons.apartment_outlined,
            label: 'Hostel Name',
            value: _hostelName,
          ),
          _darkDivider(),
          _infoRow(
            icon: Icons.category_outlined,
            label: 'Hostel Type',
            value: _typeText,
          ),
          _darkDivider(),
          _infoRow(
            icon: Icons.location_city_outlined,
            label: 'City',
            value: _city.isEmpty ? 'Not added' : _city,
          ),
          _darkDivider(),
          _infoRow(
            icon: Icons.location_on_outlined,
            label: 'Address',
            value: _address.isEmpty ? 'Address not added' : _address,
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // CAPACITY
  // ===========================================================================

  Widget _buildCapacityCard() {
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
                      'Bed Occupancy',
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontWeight: FontWeight.w800,
                        fontSize: 12,
                      ),
                    ),

                    const SizedBox(height: 3),

                    Text(
                      'Current hostel utilization',
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.48),
                        fontSize: 8.5,
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
                  color: AppTheme.teal.withOpacity(0.18),
                  borderRadius: BorderRadius.circular(18),
                ),
                child: Text(
                  '$_occupancyPercent%',
                  style: GoogleFonts.dmSans(
                    color: AppTheme.tealMint,
                    fontWeight: FontWeight.w800,
                    fontSize: 10,
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

          const SizedBox(height: 16),

          Row(
            children: [
              Expanded(
                child: _capacityBox(
                  value: '$_occupiedBeds',
                  label: 'Occupied Beds',
                  icon: Icons.groups_rounded,
                ),
              ),

              const SizedBox(width: 9),

              Expanded(
                child: _capacityBox(
                  value: '$_availableBeds',
                  label: 'Available Beds',
                  icon: Icons.event_available_rounded,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _capacityBox({
    required String value,
    required String label,
    required IconData icon,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 13),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.18),
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: AppTheme.tealMint.withOpacity(0.12)),
      ),
      child: Row(
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.18),
              borderRadius: BorderRadius.circular(11),
            ),
            child: Icon(icon, size: 17, color: AppTheme.tealMint),
          ),

          const SizedBox(width: 9),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  value,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontWeight: FontWeight.w800,
                    fontSize: 15,
                  ),
                ),
                Text(
                  label,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white.withOpacity(0.46),
                    fontSize: 8,
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
  // AMENITIES
  // ===========================================================================

  Widget _buildAmenities() {
    return LayoutBuilder(
      builder: (context, constraints) {
        final columns = constraints.maxWidth < 330 ? 2 : 3;

        const spacing = 9.0;

        final width =
            (constraints.maxWidth - ((columns - 1) * spacing)) / columns;

        return Wrap(
          spacing: spacing,
          runSpacing: spacing,
          children: _amenities.map((item) {
            final color = item['color'] as Color;

            return SizedBox(
              width: width,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  vertical: 13,
                  horizontal: 7,
                ),
                decoration: BoxDecoration(
                  color: const Color(0xD909292B),
                  borderRadius: BorderRadius.circular(17),
                  border: Border.all(color: Colors.white.withOpacity(0.09)),
                ),
                child: Column(
                  children: [
                    Container(
                      width: 39,
                      height: 39,
                      decoration: BoxDecoration(
                        color: color.withOpacity(0.13),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Icon(
                        item['icon'] as IconData,
                        size: 19,
                        color: color,
                      ),
                    ),

                    const SizedBox(height: 8),

                    Text(
                      item['name'].toString(),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      textAlign: TextAlign.center,
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 9.5,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            );
          }).toList(),
        );
      },
    );
  }

  // ===========================================================================
  // RULES
  // ===========================================================================

  Widget _buildRulesCard() {
    final visibleRules = _rules.take(3).toList();

    return _glassCard(
      child: Column(
        children: [
          ...List.generate(visibleRules.length, (index) {
            return Padding(
              padding: EdgeInsets.only(
                bottom: index < visibleRules.length - 1 ? 11 : 0,
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: 27,
                    height: 27,
                    decoration: BoxDecoration(
                      color: AppTheme.teal.withOpacity(0.18),
                      shape: BoxShape.circle,
                    ),
                    child: Center(
                      child: Text(
                        '${index + 1}',
                        style: GoogleFonts.dmSans(
                          color: AppTheme.tealMint,
                          fontSize: 9,
                          fontWeight: FontWeight.w800,
                        ),
                      ),
                    ),
                  ),

                  const SizedBox(width: 10),

                  Expanded(
                    child: Text(
                      visibleRules[index],
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.76),
                        fontSize: 10,
                        height: 1.45,
                      ),
                    ),
                  ),
                ],
              ),
            );
          }),

          if (_rules.length > 3) ...[
            const SizedBox(height: 15),

            SizedBox(
              width: double.infinity,
              height: 44,
              child: OutlinedButton.icon(
                onPressed: _showAllRules,
                icon: const Icon(Icons.visibility_outlined, size: 16),
                label: Text('View All ${_rules.length} Rules'),
                style: OutlinedButton.styleFrom(
                  foregroundColor: AppTheme.tealMint,
                  side: BorderSide(color: AppTheme.tealMint.withOpacity(0.30)),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(13),
                  ),
                ),
              ),
            ),
          ],
        ],
      ),
    );
  }

  // ===========================================================================
  // CONTACT
  // ===========================================================================

  Widget _buildContactCard() {
    return _glassCard(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 3),
      child: Column(
        children: [
          _infoRow(
            icon: Icons.person_outline_rounded,
            label: 'Owner',
            value: widget.ownerName,
          ),

          _darkDivider(),

          _infoRow(
            icon: Icons.alternate_email_rounded,
            label: 'Owner Email',
            value: widget.ownerEmail,
          ),

          _darkDivider(),

          _infoRow(
            icon: Icons.phone_outlined,
            label: 'Hostel Phone',
            value: _phone,
          ),

          _darkDivider(),

          _infoRow(
            icon: Icons.email_outlined,
            label: 'Hostel Email',
            value: _email,
          ),
        ],
      ),
    );
  }

  // ===========================================================================
  // SHARED COMPONENTS
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

  Widget _infoRow({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Container(
            width: 39,
            height: 39,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(0.17),
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
                    color: Colors.white.withOpacity(0.43),
                    fontSize: 8.5,
                  ),
                ),

                const SizedBox(height: 3),

                Text(
                  value,
                  maxLines: 3,
                  overflow: TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 10.5,
                    fontWeight: FontWeight.w600,
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _darkDivider() {
    return Divider(
      height: 1,
      thickness: 0.7,
      color: Colors.white.withOpacity(0.07),
    );
  }

  // ===========================================================================
  // EDIT HOSTEL
  // ===========================================================================

  Future<void> _openEditHostel() async {
    await _openEditor();
  }

  Future<void> _openEditor({String initialSection = 'details'}) async {
    final result = await Navigator.push<Map<String, dynamic>>(
      context,
      MaterialPageRoute(
        builder: (_) => HostelProfileEditorScreen(
          hostelName: _hostelName,
          initialSection: initialSection,
        ),
      ),
    );

    if (result == null) {
      return;
    }

    setState(() {
      _hostelName = result['hostelName']?.toString() ?? _hostelName;

      _hostelType = result['hostelType']?.toString() ?? _hostelType;

      _city = result['city']?.toString() ?? _city;

      _address = result['address']?.toString() ?? _address;

      _phone = result['phone']?.toString() ?? _phone;

      _email = result['email']?.toString() ?? _email;
    });

    if (!mounted) return;
    await _loadHostelData();
    if (!mounted) return;
    _showMessage('Hostel information database mein update ho gayi.');
  }

  // ===========================================================================
  // RULES SHEET
  // ===========================================================================

  void _showAllRules() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (sheetContext) {
        return SafeArea(
          top: false,
          child: Container(
            constraints: BoxConstraints(
              maxHeight: MediaQuery.of(context).size.height * 0.78,
            ),
            margin: const EdgeInsets.all(12),
            padding: const EdgeInsets.fromLTRB(18, 12, 18, 18),
            decoration: BoxDecoration(
              color: const Color(0xFF082C2F),
              borderRadius: BorderRadius.circular(28),
              border: Border.all(color: AppTheme.tealMint.withOpacity(0.18)),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Container(
                    width: 42,
                    height: 4,
                    margin: const EdgeInsets.only(bottom: 18),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.20),
                      borderRadius: BorderRadius.circular(3),
                    ),
                  ),
                ),

                Text(
                  'Hostel Rules',
                  style: GoogleFonts.playfairDisplay(
                    color: Colors.white,
                    fontSize: 23,
                    fontWeight: FontWeight.w700,
                  ),
                ),

                const SizedBox(height: 4),

                Text(
                  '${_rules.length} active hostel policies',
                  style: GoogleFonts.dmSans(
                    color: Colors.white.withOpacity(0.50),
                    fontSize: 9.5,
                  ),
                ),

                const SizedBox(height: 16),

                Flexible(
                  child: SingleChildScrollView(
                    physics: const BouncingScrollPhysics(),
                    child: Column(
                      children: List.generate(_rules.length, (index) {
                        return Container(
                          width: double.infinity,
                          margin: const EdgeInsets.only(bottom: 8),
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: Colors.black.withOpacity(0.19),
                            borderRadius: BorderRadius.circular(14),
                            border: Border.all(
                              color: Colors.white.withOpacity(0.08),
                            ),
                          ),
                          child: Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Container(
                                width: 26,
                                height: 26,
                                decoration: BoxDecoration(
                                  color: AppTheme.teal.withOpacity(0.18),
                                  shape: BoxShape.circle,
                                ),
                                child: Center(
                                  child: Text(
                                    '${index + 1}',
                                    style: GoogleFonts.dmSans(
                                      color: AppTheme.tealMint,
                                      fontWeight: FontWeight.w800,
                                      fontSize: 9,
                                    ),
                                  ),
                                ),
                              ),

                              const SizedBox(width: 10),

                              Expanded(
                                child: Text(
                                  _rules[index],
                                  style: GoogleFonts.dmSans(
                                    color: Colors.white.withOpacity(0.75),
                                    fontSize: 10,
                                    height: 1.45,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        );
                      }),
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
            padding: const EdgeInsets.all(16),
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
                    width: 38,
                    height: 4,
                    margin: const EdgeInsets.only(bottom: 12),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.18),
                      borderRadius: BorderRadius.circular(3),
                    ),
                  ),
                ),

                _menuTile(
                  icon: Icons.edit_outlined,
                  title: 'Edit Hostel',
                  subtitle: 'Update hostel information',
                  onTap: () {
                    Navigator.pop(sheetContext);

                    _openEditHostel();
                  },
                ),

                _menuTile(
                  icon: Icons.add_photo_alternate_outlined,
                  title: 'Hostel Photos',
                  subtitle: 'Manage hostel images and gallery',
                  onTap: () {
                    Navigator.pop(sheetContext);
                    _openEditor(initialSection: 'photos');
                  },
                ),

                _menuTile(
                  icon: Icons.location_on_outlined,
                  title: 'Location & Map',
                  subtitle: 'Update hostel map location',
                  onTap: () {
                    Navigator.pop(sheetContext);
                    _openEditor(initialSection: 'location');
                  },
                ),

                _menuTile(
                  icon: _isActive
                      ? Icons.pause_circle_outline_rounded
                      : Icons.play_circle_outline_rounded,
                  title: _isActive ? 'Deactivate Hostel' : 'Activate Hostel',
                  subtitle: _isActive
                      ? 'Temporarily disable hostel listing'
                      : 'Restore hostel listing',
                  onTap: () {
                    Navigator.pop(sheetContext);

                    _confirmStatusChange();
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
                        color: Colors.white.withOpacity(0.48),
                        fontSize: 8.5,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(width: 6),

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
  // STATUS CHANGE
  // ===========================================================================

  void _confirmStatusChange() {
    showDialog(
      context: context,
      builder: (dialogContext) {
        return AlertDialog(
          backgroundColor: const Color(0xFF082C2F),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(22),
          ),
          title: Text(
            _isActive ? 'Deactivate Hostel?' : 'Activate Hostel?',
            style: GoogleFonts.playfairDisplay(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.w700,
            ),
          ),
          content: Text(
            _isActive
                ? 'The hostel will be marked inactive. Existing records will remain available.'
                : 'The hostel will become active again.',
            style: GoogleFonts.dmSans(
              color: Colors.white.withOpacity(0.60),
              fontSize: 10.5,
              height: 1.5,
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
                  color: Colors.white.withOpacity(0.55),
                ),
              ),
            ),

            ElevatedButton(
              onPressed: () async {
                Navigator.pop(dialogContext);
                final hostelId = _hostelId;
                if (hostelId == null) {
                  _showMessage('Hostel id unavailable. Please reload.');
                  return;
                }
                try {
                  final nextActive = !_isActive;
                  await Api().updateOwnerHostel(
                    hostelId: hostelId,
                    updates: {
                      'publication_status': nextActive
                          ? 'published'
                          : 'suspended',
                    },
                  );
                  if (!mounted) return;
                  setState(() => _isActive = nextActive);
                  _showMessage(
                    nextActive
                        ? 'Hostel listing activated.'
                        : 'Hostel listing deactivated.',
                  );
                } catch (error) {
                  if (!mounted) return;
                  _showMessage('Status update failed: $error');
                }
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: _isActive
                    ? const Color(0xFFD55353)
                    : AppTheme.teal,
                foregroundColor: Colors.white,
                elevation: 0,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(13),
                ),
              ),
              child: Text(_isActive ? 'Deactivate' : 'Activate'),
            ),
          ],
        );
      },
    );
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
            borderRadius: BorderRadius.circular(14),
          ),
          content: Row(
            children: [
              const Icon(
                Icons.check_circle_outline_rounded,
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
      );
  }
}
