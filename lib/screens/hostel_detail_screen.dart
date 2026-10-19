import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import '../api.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'student_booking_info_screen.dart';
import 'my_bookings_screen.dart';

class HostelDetailScreen extends StatefulWidget {
  final int id;
  const HostelDetailScreen({super.key, required this.id});

  @override
  State<HostelDetailScreen> createState() => _HostelDetailScreenState();
}

class _HostelDetailScreenState extends State<HostelDetailScreen> {
  final api = Api();
  Map<String, dynamic>? hostel;
  String? _error;
  bool _loading = true;
  bool _isFav = false;

  Future<void> _openBookingFlow() async {
    try {
      final bookings = await api.getBookings();
      final alreadyRequested = bookings.whereType<Map>().any((raw) {
        final booking = Map<String, dynamic>.from(raw);
        final status = booking['status']?.toString().toLowerCase();
        return booking['hostel_id']?.toString() == widget.id.toString() &&
            (status == 'pending' || status == 'confirmed');
      });
      if (!mounted) return;
      if (alreadyRequested) {
        Navigator.push(
          context,
          MaterialPageRoute(builder: (_) => const MyBookingsScreen()),
        );
        return;
      }
    } catch (_) {
      // A failed duplicate check must not prevent a new student from booking.
    }
    if (!mounted) return;
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => StudentBookingInfoScreen(
          hostelId: widget.id,
          hostelName: hostel!['name'] ?? 'Hostel',
          hostelType: hostel!['hostel_type'] ?? '',
          city: hostel!['city'] ?? '',
          address: hostel!['address'] ?? '',
          pricePerMonth: int.tryParse('${hostel!['single_room_price'] ?? 0}') ?? 0,
          distanceKm: 0,
          rating: double.tryParse('${hostel!['average_rating'] ?? ''}'),
        ),
      ),
    );
  }

  @override
  void initState() {
    super.initState();
    _loadHostel();
  }

  Future<void> _loadHostel() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final data = await api
          .getHostelDetail(widget.id)
          .timeout(const Duration(seconds: 12));
      if (!mounted) return;
      setState(() {
        hostel = data;
        _loading = false;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        _loading = false;
        _error = error.toString().replaceFirst('Exception: ', '');
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: .48,
          tintColor: AppTheme.tealDeep,
          tintOpacity: .28,
          cardMargin: const EdgeInsets.all(12),
          cardRadius: 30,
          cardColor: AppTheme.bg.withOpacity(.97),
          child: _loading
              ? const Center(
                  child: CircularProgressIndicator(color: AppTheme.teal),
                )
              : _error != null
              ? _errorState()
              : _buildContent(context),
        ),
      ),
    );
  }

  Widget _errorState() => Center(
    child: Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 70,
            height: 70,
            decoration: BoxDecoration(
              color: AppTheme.teal.withOpacity(.10),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.wifi_off_rounded,
              color: AppTheme.teal,
              size: 32,
            ),
          ),
          const SizedBox(height: 14),
          Text(
            'Hostel details unavailable',
            style: GoogleFonts.playfairDisplay(
              fontSize: 22,
              fontWeight: FontWeight.w700,
              color: AppTheme.textPrimary,
            ),
          ),
          const SizedBox(height: 7),
          Text(
            _error ?? 'Unable to load this published hostel.',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              fontSize: 12,
              color: AppTheme.textSecondary,
            ),
          ),
          const SizedBox(height: 16),
          FilledButton.icon(
            onPressed: _loadHostel,
            icon: const Icon(Icons.refresh_rounded),
            label: const Text('Try again'),
          ),
        ],
      ),
    ),
  );

  Widget _buildContent(BuildContext context) {
    final name = hostel!['name'] ?? 'Hostel';
    final city = hostel!['city'] ?? '';
    final address = hostel!['address'] ?? '';
    final description = hostel!['description'] ?? '';
    final avail = hostel!['available_capacity'] ?? 0;
    final total = hostel!['total_capacity'] ?? 0;
    final reviews = (hostel!['reviews'] ?? []) as List;

    return Column(
      children: [
        Expanded(
          child: CustomScrollView(
            slivers: [
              // Hero
              SliverToBoxAdapter(child: _buildHero(context, name, city)),

              // Content
              SliverPadding(
                padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
                sliver: SliverList(
                  delegate: SliverChildListDelegate([
                    // Address
                    if (address.isNotEmpty) ...[
                      Row(
                        children: [
                          Icon(
                            Icons.location_on_rounded,
                            size: 14,
                            color: AppTheme.textMuted,
                          ),
                          const SizedBox(width: 4),
                          Expanded(
                            child: Text(
                              address,
                              style: GoogleFonts.dmSans(
                                fontSize: 12,
                                color: AppTheme.textMuted,
                              ),
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 20),
                    ],

                    // Stat cards
                    _buildStatGrid(avail, total),
                    const SizedBox(height: 24),

                    // Prices and room types come directly from the published
                    // hostel record; do not show invented room prices.
                    _buildPricing(),
                    const SizedBox(height: 24),

                    // Description
                    if (description.isNotEmpty) ...[
                      _sectionLabel('About'),
                      const SizedBox(height: 8),
                      Text(
                        description,
                        style: GoogleFonts.dmSans(
                          fontSize: 14,
                          color: AppTheme.textSecondary,
                          height: 1.55,
                        ),
                      ),
                      const SizedBox(height: 24),
                    ],

                    // Amenities (static showcase)
                    _sectionLabel('Amenities'),
                    const SizedBox(height: 10),
                    _buildAmenities(),
                    const SizedBox(height: 24),

                    _buildRulesAndMess(),
                    const SizedBox(height: 24),

                    // Reviews
                    _sectionLabel('Reviews (${reviews.length})'),
                    const SizedBox(height: 12),
                    ...reviews.map((r) => _ReviewCard(review: r)),

                    if (reviews.isEmpty)
                      Container(
                        padding: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          color: AppTheme.card,
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(
                            color: AppTheme.border,
                            width: 0.5,
                          ),
                        ),
                        child: Center(
                          child: Text(
                            'No reviews yet.',
                            style: GoogleFonts.dmSans(
                              color: AppTheme.textMuted,
                              fontSize: 13,
                            ),
                          ),
                        ),
                      ),

                    const SizedBox(height: 20),
                  ]),
                ),
              ),
            ],
          ),
        ),

        // Sticky bottom bar
        _buildBottomBar(),
      ],
    );
  }

  Widget _buildHero(BuildContext context, String name, String city) {
    return Stack(
      children: [
        Container(
          height: 230,
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
              colors: [Color(0xFF1D9E75), AppTheme.teal, AppTheme.tealDeep],
            ),
          ),
          child: SafeArea(
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 70,
                    height: 70,
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.15),
                      shape: BoxShape.circle,
                    ),
                    child: const Icon(
                      Icons.business_rounded,
                      size: 34,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    name,
                    style: GoogleFonts.playfairDisplay(
                      fontSize: 22,
                      fontWeight: FontWeight.w700,
                      color: Colors.white,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 4),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(
                        Icons.location_on_rounded,
                        size: 14,
                        color: Colors.white70,
                      ),
                      const SizedBox(width: 3),
                      Text(
                        city,
                        style: GoogleFonts.dmSans(
                          fontSize: 13,
                          color: Colors.white.withOpacity(0.75),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ),

        // Back button
        Positioned(
          top: MediaQuery.of(context).padding.top + 8,
          left: 12,
          child: _heroButton(
            child: const Icon(
              Icons.arrow_back_rounded,
              size: 20,
              color: Colors.white,
            ),
            onTap: () => Navigator.pop(context),
          ),
        ),

        // Favourite button
        Positioned(
          top: MediaQuery.of(context).padding.top + 8,
          right: 12,
          child: _heroButton(
            child: Icon(
              _isFav ? Icons.favorite_rounded : Icons.favorite_border_rounded,
              size: 20,
              color: _isFav ? Colors.redAccent : Colors.white,
            ),
            onTap: () => setState(() => _isFav = !_isFav),
          ),
        ),
      ],
    );
  }

  Widget _heroButton({required Widget child, required VoidCallback onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 38,
        height: 38,
        decoration: BoxDecoration(
          color: Colors.black.withOpacity(0.22),
          shape: BoxShape.circle,
          border: Border.all(color: Colors.white.withOpacity(0.2)),
        ),
        child: Center(child: child),
      ),
    );
  }

  Widget _buildStatGrid(int avail, int total) {
    final occupancy = total > 0 ? ((total - avail) / total * 100).round() : 0;
    return Row(
      children: [
        _StatCard(value: '$avail', label: 'Available', icon: Icons.bed_rounded),
        const SizedBox(width: 10),
        _StatCard(
          value: '$total',
          label: 'Capacity',
          icon: Icons.people_rounded,
        ),
        const SizedBox(width: 10),
        _StatCard(
          value: '$occupancy%',
          label: 'Occupied',
          icon: Icons.pie_chart_rounded,
          highlight: occupancy > 80,
        ),
      ],
    );
  }

  Widget _sectionLabel(String text) {
    return Text(
      text,
      style: GoogleFonts.dmSans(
        fontSize: 15,
        fontWeight: FontWeight.w700,
        color: AppTheme.textPrimary,
      ),
    );
  }

  Widget _buildAmenities() {
    final names = <String>{
      if (hostel?['amenities'] is List)
        ...(hostel!['amenities'] as List).map((e) => e.toString()),
      if (hostel?['facilities'] is List)
        ...(hostel!['facilities'] as List).map((e) => e.toString()),
    }.where((e) => e.trim().isNotEmpty).toList();
    if (names.isEmpty) {
      return Text(
        'No amenities listed yet.',
        style: GoogleFonts.dmSans(color: AppTheme.textMuted, fontSize: 13),
      );
    }
    IconData iconFor(String name) {
      final value = name.toLowerCase();
      if (value.contains('wifi') || value.contains('internet'))
        return Icons.wifi_rounded;
      if (value.contains('security') || value.contains('guard'))
        return Icons.security_rounded;
      if (value.contains('parking')) return Icons.local_parking_rounded;
      if (value.contains('mess') ||
          value.contains('food') ||
          value.contains('meal'))
        return Icons.restaurant_rounded;
      if (value.contains('laundry')) return Icons.local_laundry_service_rounded;
      if (value.contains('cctv') || value.contains('camera'))
        return Icons.videocam_rounded;
      if (value.contains('water')) return Icons.water_drop_rounded;
      if (value.contains('power') || value.contains('generator'))
        return Icons.bolt_rounded;
      return Icons.check_circle_outline_rounded;
    }

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: names.map((name) {
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 7),
          decoration: BoxDecoration(
            color: AppTheme.teal.withOpacity(0.07),
            borderRadius: BorderRadius.circular(10),
            border: Border.all(
              color: AppTheme.teal.withOpacity(0.18),
              width: 0.5,
            ),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(iconFor(name), size: 15, color: AppTheme.teal),
              const SizedBox(width: 5),
              Text(
                name,
                style: GoogleFonts.dmSans(
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                  color: AppTheme.teal,
                ),
              ),
            ],
          ),
        );
      }).toList(),
    );
  }

  Widget _buildPricing() {
    final prices =
        <Map<String, dynamic>>[
              {'label': 'Single room', 'value': hostel?['single_room_price']},
              {'label': 'Double room', 'value': hostel?['double_room_price']},
              {'label': 'Dorm room', 'value': hostel?['dorm_room_price']},
            ]
            .where(
              (item) =>
                  item['value'] != null && item['value'].toString() != '0',
            )
            .toList();
    final types = hostel?['room_types_available'];
    final typeNames = types is List
        ? types
              .map((e) => e.toString())
              .where((e) => e.trim().isNotEmpty)
              .toList()
        : <String>[];
    if (prices.isEmpty && typeNames.isEmpty) return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        _sectionLabel('Rent & Room Options'),
        const SizedBox(height: 10),
        if (prices.isNotEmpty)
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: prices
                .map(
                  (item) => Container(
                    padding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 10,
                    ),
                    decoration: BoxDecoration(
                      color: AppTheme.card,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: AppTheme.border),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          item['label'].toString(),
                          style: GoogleFonts.dmSans(
                            fontSize: 11,
                            color: AppTheme.textMuted,
                          ),
                        ),
                        Text(
                          'PKR ${item['value']} / month',
                          style: GoogleFonts.dmSans(
                            fontSize: 12,
                            fontWeight: FontWeight.w700,
                            color: AppTheme.tealDeep,
                          ),
                        ),
                      ],
                    ),
                  ),
                )
                .toList(),
          ),
        if (typeNames.isNotEmpty) ...[
          const SizedBox(height: 10),
          Text(
            'Available types: ${typeNames.join(' · ')}',
            style: GoogleFonts.dmSans(
              fontSize: 12,
              color: AppTheme.textSecondary,
            ),
          ),
        ],
      ],
    );
  }

  Widget _buildRulesAndMess() {
    final rules = hostel?['rules'];
    final mess = hostel?['mess_info'];
    final ruleNames = rules is List
        ? rules
              .map((e) => e.toString())
              .where((e) => e.trim().isNotEmpty)
              .toList()
        : <String>[];
    final messText = mess is Map
        ? mess.entries.map((e) => '${e.key}: ${e.value}').join(' · ')
        : (mess?.toString() ?? '');
    if (ruleNames.isEmpty && messText.trim().isEmpty)
      return const SizedBox.shrink();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (messText.trim().isNotEmpty) ...[
          _sectionLabel('Food & Mess'),
          const SizedBox(height: 8),
          _infoPanel(Icons.restaurant_rounded, messText),
          const SizedBox(height: 18),
        ],
        if (ruleNames.isNotEmpty) ...[
          _sectionLabel('Hostel Rules'),
          const SizedBox(height: 8),
          ...ruleNames.map(
            (rule) => Padding(
              padding: const EdgeInsets.only(bottom: 7),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Icon(
                    Icons.check_circle_outline_rounded,
                    size: 16,
                    color: AppTheme.teal,
                  ),
                  const SizedBox(width: 7),
                  Expanded(
                    child: Text(
                      rule,
                      style: GoogleFonts.dmSans(
                        fontSize: 12,
                        color: AppTheme.textSecondary,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ],
    );
  }

  Widget _infoPanel(IconData icon, String text) => Container(
    width: double.infinity,
    padding: const EdgeInsets.all(13),
    decoration: BoxDecoration(
      color: AppTheme.teal.withOpacity(.06),
      borderRadius: BorderRadius.circular(13),
      border: Border.all(color: AppTheme.teal.withOpacity(.15)),
    ),
    child: Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 17, color: AppTheme.teal),
        const SizedBox(width: 8),
        Expanded(
          child: Text(
            text,
            style: GoogleFonts.dmSans(
              fontSize: 12,
              color: AppTheme.textSecondary,
              height: 1.4,
            ),
          ),
        ),
      ],
    ),
  );

  Widget _buildBottomBar() {
    return Container(
      padding: EdgeInsets.fromLTRB(
        20,
        14,
        20,
        14 + MediaQuery.of(context).padding.bottom,
      ),
      decoration: BoxDecoration(
        color: AppTheme.card,
        border: Border(top: BorderSide(color: AppTheme.border, width: 0.5)),
      ),
      child: Row(
        children: [
          Expanded(
            child: SizedBox(
              height: 50,
              child: ElevatedButton(
                onPressed: _openBookingFlow,
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
                    const Icon(Icons.hotel_rounded, size: 18),
                    const SizedBox(width: 8),
                    Text(
                      'Book Now',
                      style: GoogleFonts.dmSans(
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Container(
            width: 50,
            height: 50,
            decoration: BoxDecoration(
              color: AppTheme.bgSecondary,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: AppTheme.border, width: 0.5),
            ),
            child: IconButton(
              onPressed: () {},
              icon: const Icon(
                Icons.share_rounded,
                size: 20,
                color: AppTheme.textSecondary,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _StatCard extends StatelessWidget {
  final String value;
  final String label;
  final IconData icon;
  final bool highlight;

  const _StatCard({
    required this.value,
    required this.label,
    required this.icon,
    this.highlight = false,
  });

  @override
  Widget build(BuildContext context) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 10),
        decoration: BoxDecoration(
          color: AppTheme.card,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(
            color: highlight ? AppTheme.teal.withOpacity(0.3) : AppTheme.border,
            width: 0.5,
          ),
        ),
        child: Column(
          children: [
            Icon(icon, size: 20, color: AppTheme.teal),
            const SizedBox(height: 6),
            Text(
              value,
              style: GoogleFonts.dmSans(
                fontSize: 18,
                fontWeight: FontWeight.w700,
                color: AppTheme.teal,
              ),
            ),
            Text(
              label,
              style: GoogleFonts.dmSans(
                fontSize: 11,
                color: AppTheme.textMuted,
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ReviewCard extends StatelessWidget {
  final dynamic review;
  const _ReviewCard({required this.review});

  @override
  Widget build(BuildContext context) {
    final rating = (review['overall_rating'] ?? 0) as num;
    final text = review['text_review'] ?? '';

    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: AppTheme.card,
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: AppTheme.border, width: 0.5),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Stars
            Row(
              children: [
                ...List.generate(5, (i) {
                  return Icon(
                    i < rating.round()
                        ? Icons.star_rounded
                        : Icons.star_outline_rounded,
                    size: 16,
                    color: const Color(0xFFEF9F27),
                  );
                }),
                const SizedBox(width: 8),
                Text(
                  '$rating / 5',
                  style: GoogleFonts.dmSans(
                    fontSize: 12,
                    fontWeight: FontWeight.w600,
                    color: const Color(0xFFBA7517),
                  ),
                ),
              ],
            ),
            if (text.isNotEmpty) ...[
              const SizedBox(height: 8),
              Text(
                text,
                style: GoogleFonts.dmSans(
                  fontSize: 13,
                  color: AppTheme.textSecondary,
                  height: 1.5,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}
