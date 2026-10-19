import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../models/owner_room_setup_draft.dart';
import '../theme.dart';
import '../widgets/responsive_shell.dart';
import '../widgets/video_background.dart';
import 'owner_step_facilities.dart';

class OwnerFloorRoomSetupScreen extends StatefulWidget {
  final String ownerEmail;
  final String ownerName;
  final String hostelName;
  final String hostelType;
  final String city;
  final String address;
  final double latitude;
  final double longitude;

  const OwnerFloorRoomSetupScreen({
    super.key,
    required this.ownerEmail,
    required this.ownerName,
    required this.hostelName,
    required this.hostelType,
    required this.city,
    required this.address,
    required this.latitude,
    required this.longitude,
  });

  @override
  State<OwnerFloorRoomSetupScreen> createState() =>
      _OwnerFloorRoomSetupScreenState();
}

class _OwnerFloorRoomSetupScreenState extends State<OwnerFloorRoomSetupScreen> {
  final List<_FloorDraft> _floors = [_FloorDraft.create(1)];
  String? _error;

  @override
  void dispose() {
    for (final floor in _floors) {
      floor.dispose();
    }
    super.dispose();
  }

  void _changeFloorCount(int change) {
    setState(() {
      _error = null;
      if (change > 0 && _floors.length < 20) {
        _floors.add(_FloorDraft.create(_floors.length + 1));
      } else if (change < 0 && _floors.length > 1) {
        _floors.removeLast().dispose();
      }
    });
  }

  void _addRoom(_FloorDraft floor) {
    setState(() {
      _error = null;
      floor.rooms.add(_RoomDraft.create(floor.rooms.length + 1));
    });
  }

  void _removeRoom(_FloorDraft floor, _RoomDraft room) {
    if (floor.rooms.length == 1) {
      setState(() => _error = 'Every floor needs at least one room.');
      return;
    }
    setState(() {
      _error = null;
      floor.rooms.remove(room);
      room.dispose();
    });
  }

  void _setCapacity(_RoomDraft room, int capacity) {
    setState(() {
      room.capacity = capacity;
      while (room.bedControllers.length < capacity) {
        room.bedControllers.add(
          TextEditingController(text: '${room.bedControllers.length + 1}'),
        );
      }
      while (room.bedControllers.length > capacity) {
        room.bedControllers.removeLast().dispose();
      }
    });
  }

  void _next() {
    FocusScope.of(context).unfocus();
    final plannedRooms = <Map<String, dynamic>>[];
    final seenRoomNumbers = <String>{};

    for (var floorIndex = 0; floorIndex < _floors.length; floorIndex++) {
      final floor = _floors[floorIndex];
      final floorLabel = floor.labelController.text.trim();
      if (floorLabel.isEmpty) {
        setState(
          () =>
              _error = 'Enter a floor number/name for floor ${floorIndex + 1}.',
        );
        return;
      }
      if (floor.rooms.isEmpty) {
        setState(() => _error = '$floorLabel needs at least one room.');
        return;
      }
      for (final room in floor.rooms) {
        final roomNumber = room.numberController.text.trim();
        final rent = int.tryParse(room.rentController.text.trim());
        final roomKey = roomNumber.toLowerCase();
        final beds = room.bedControllers
            .map((controller) => controller.text.trim())
            .toList();
        if (roomNumber.isEmpty) {
          setState(() => _error = 'Enter a room number on $floorLabel.');
          return;
        }
        if (!seenRoomNumbers.add(roomKey)) {
          setState(
            () => _error = 'Room $roomNumber is entered more than once.',
          );
          return;
        }
        if (rent == null || rent < 0) {
          setState(
            () => _error = 'Enter a valid monthly rent for room $roomNumber.',
          );
          return;
        }
        if (beds.length != room.capacity || beds.any((bed) => bed.isEmpty)) {
          setState(
            () => _error =
                'Enter all ${room.capacity} bed numbers for room $roomNumber.',
          );
          return;
        }
        if (beds.map((bed) => bed.toLowerCase()).toSet().length !=
            beds.length) {
          setState(
            () => _error = 'Bed numbers must be unique in room $roomNumber.',
          );
          return;
        }
        plannedRooms.add({
          'floor_label': floorLabel,
          'room_number': roomNumber,
          'capacity': room.capacity,
          'room_type': '${room.capacity} Seater',
          'monthly_rent': rent,
          'beds': beds,
        });
      }
    }

    final totalCapacity = plannedRooms.fold<int>(
      0,
      (sum, room) => sum + room['capacity'] as int,
    );
    final roomTypes = plannedRooms
        .map(
          (room) => <String, dynamic>{
            'type': room['room_type'],
            'price': room['monthly_rent'],
          },
        )
        .toList();
    OwnerRoomSetupDraft.replace(plannedRooms);
    setState(() => _error = null);

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => OwnerStepFacilities(
          ownerEmail: widget.ownerEmail,
          ownerName: widget.ownerName,
          hostelName: widget.hostelName,
          hostelType: widget.hostelType,
          city: widget.city,
          address: widget.address,
          latitude: widget.latitude,
          longitude: widget.longitude,
          totalFloors: _floors.length,
          totalRooms: plannedRooms.length,
          totalCapacity: totalCapacity,
          roomTypes: roomTypes,
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final keyboard = MediaQuery.viewInsetsOf(context).bottom;
    return ResponsiveShell(
      child: Scaffold(
        backgroundColor: Colors.transparent,
        body: VideoBackground(
          assetPath: 'assets/images/background.mp4',
          overlayOpacity: 0.28,
          cardMargin: const EdgeInsets.all(14),
          cardRadius: 32,
          child: SafeArea(
            child: SingleChildScrollView(
              keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
              padding: EdgeInsets.fromLTRB(18, 12, 18, 24 + keyboard),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [_header(), const SizedBox(height: 18), _card()],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _header() => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Row(
        children: [
          _roundIcon(
            Icons.arrow_back_rounded,
            () => Navigator.maybePop(context),
          ),
          const Spacer(),
          _pill('Step 5 of 9'),
        ],
      ),
      const SizedBox(height: 16),
      Row(
        children: List.generate(
          9,
          (index) => Expanded(
            child: Container(
              height: index < 5 ? 4 : 3,
              margin: EdgeInsets.only(right: index < 8 ? 5 : 0),
              decoration: BoxDecoration(
                color: index < 5
                    ? AppTheme.tealMint
                    : Colors.white.withOpacity(.22),
                borderRadius: BorderRadius.circular(4),
              ),
            ),
          ),
        ),
      ),
      const SizedBox(height: 16),
      Text(
        'Floor, Room\n& Bed Details',
        style: GoogleFonts.playfairDisplay(
          color: Colors.white,
          fontSize: 34,
          height: 1.03,
          fontWeight: FontWeight.w700,
        ),
      ),
      const SizedBox(height: 10),
      Text(
        'Set rooms floor-by-floor. Each room can have 1 to 6 named beds.',
        style: GoogleFonts.dmSans(
          color: Colors.white.withOpacity(.72),
          fontSize: 12.5,
          height: 1.4,
        ),
      ),
    ],
  );

  Widget _card() => Container(
    width: double.infinity,
    padding: const EdgeInsets.fromLTRB(16, 16, 16, 20),
    decoration: BoxDecoration(
      color: const Color(0xFF023336).withOpacity(.92),
      borderRadius: BorderRadius.circular(28),
      border: Border.all(color: AppTheme.tealMint.withOpacity(.22)),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Center(
          child: Container(
            width: 38,
            height: 4,
            margin: const EdgeInsets.only(bottom: 16),
            decoration: BoxDecoration(
              color: AppTheme.tealMint.withOpacity(.3),
              borderRadius: BorderRadius.circular(4),
            ),
          ),
        ),
        _sectionTitle(Icons.layers_rounded, 'How many floors?'),
        const SizedBox(height: 8),
        _floorCountControl(),
        const SizedBox(height: 9),
        Text(
          'Add the rooms separately for each floor. Capacity is calculated from the beds you enter.',
          style: GoogleFonts.dmSans(
            color: Colors.white.withOpacity(.58),
            fontSize: 10,
            height: 1.4,
          ),
        ),
        const SizedBox(height: 18),
        ..._floors.asMap().entries.map(
          (entry) => Padding(
            padding: const EdgeInsets.only(bottom: 14),
            child: _floorCard(entry.key, entry.value),
          ),
        ),
        if (_error != null) _errorBox(),
        const SizedBox(height: 6),
        SizedBox(
          width: double.infinity,
          height: 52,
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
                  'Next: Facilities',
                  style: GoogleFonts.dmSans(
                    fontWeight: FontWeight.w800,
                    fontSize: 13.5,
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
  );

  Widget _floorCountControl() => Container(
    height: 52,
    decoration: BoxDecoration(
      color: Colors.white.withOpacity(.95),
      borderRadius: BorderRadius.circular(15),
    ),
    child: Row(
      children: [
        _counterButton(
          Icons.remove_rounded,
          () => _changeFloorCount(-1),
          false,
        ),
        Expanded(
          child: Center(
            child: Text(
              '${_floors.length}',
              style: GoogleFonts.dmSans(
                color: const Color(0xFF183636),
                fontWeight: FontWeight.w800,
                fontSize: 16,
              ),
            ),
          ),
        ),
        _counterButton(Icons.add_rounded, () => _changeFloorCount(1), true),
      ],
    ),
  );

  Widget _counterButton(IconData icon, VoidCallback tap, bool filled) =>
      InkWell(
        onTap: tap,
        child: Container(
          width: 52,
          height: double.infinity,
          decoration: BoxDecoration(
            color: filled ? AppTheme.teal : const Color(0xFFE7F0EE),
            borderRadius: BorderRadius.horizontal(
              left: filled ? Radius.zero : const Radius.circular(15),
              right: filled ? const Radius.circular(15) : Radius.zero,
            ),
          ),
          child: Icon(icon, color: filled ? Colors.white : AppTheme.teal),
        ),
      );

  Widget _floorCard(int index, _FloorDraft floor) => Container(
    padding: const EdgeInsets.all(12),
    decoration: BoxDecoration(
      color: const Color(0xFF011F21).withOpacity(.48),
      borderRadius: BorderRadius.circular(18),
      border: Border.all(color: Colors.white.withOpacity(.12)),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: _lightField(
                floor.labelController,
                'Floor number / name',
                icon: Icons.layers_rounded,
              ),
            ),
            const SizedBox(width: 8),
            _pill(
              '${floor.rooms.length} room${floor.rooms.length == 1 ? '' : 's'}',
            ),
          ],
        ),
        const SizedBox(height: 10),
        ...floor.rooms.asMap().entries.map(
          (entry) => Padding(
            padding: const EdgeInsets.only(bottom: 10),
            child: _roomCard(floor, entry.value),
          ),
        ),
        TextButton.icon(
          onPressed: () => _addRoom(floor),
          style: TextButton.styleFrom(
            foregroundColor: AppTheme.tealMint,
            padding: const EdgeInsets.symmetric(horizontal: 4),
          ),
          icon: const Icon(Icons.add_circle_outline_rounded, size: 18),
          label: Text(
            'Add room on this floor',
            style: GoogleFonts.dmSans(
              fontWeight: FontWeight.w700,
              fontSize: 11,
            ),
          ),
        ),
      ],
    ),
  );

  Widget _roomCard(_FloorDraft floor, _RoomDraft room) => Container(
    padding: const EdgeInsets.all(10),
    decoration: BoxDecoration(
      color: AppTheme.teal.withOpacity(.11),
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: AppTheme.tealMint.withOpacity(.18)),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: _lightField(
                room.numberController,
                'Room no. e.g. 101',
                icon: Icons.meeting_room_rounded,
              ),
            ),
            IconButton(
              onPressed: () => _removeRoom(floor, room),
              tooltip: 'Remove room',
              icon: const Icon(
                Icons.delete_outline_rounded,
                color: Color(0xFFFFA0A0),
                size: 20,
              ),
            ),
          ],
        ),
        const SizedBox(height: 9),
        Row(
          children: [
            Expanded(child: _capacityPicker(room)),
            const SizedBox(width: 9),
            Expanded(
              child: _lightField(
                room.rentController,
                'Monthly rent',
                currency: true,
                digits: true,
              ),
            ),
          ],
        ),
        const SizedBox(height: 10),
        Text(
          'Bed numbers (${room.capacity})',
          style: GoogleFonts.dmSans(
            color: Colors.white.withOpacity(.78),
            fontSize: 10.5,
            fontWeight: FontWeight.w700,
          ),
        ),
        const SizedBox(height: 6),
        Wrap(
          spacing: 7,
          runSpacing: 7,
          children: [
            for (var index = 0; index < room.bedControllers.length; index++)
              SizedBox(
                width: 88,
                child: _lightField(
                  room.bedControllers[index],
                  'Bed ${index + 1}',
                  compact: true,
                ),
              ),
          ],
        ),
      ],
    ),
  );

  Widget _capacityPicker(_RoomDraft room) => DropdownButtonFormField<int>(
    value: room.capacity,
    isExpanded: true,
    dropdownColor: Colors.white,
    style: GoogleFonts.dmSans(
      color: const Color(0xFF183636),
      fontWeight: FontWeight.w700,
      fontSize: 11,
    ),
    decoration: _lightDecoration('Seater', null),
    items: [
      for (var capacity = 1; capacity <= 6; capacity++)
        DropdownMenuItem(
          value: capacity,
          child: Text(
            '$capacity Seater',
            maxLines: 1,
            overflow: TextOverflow.fade,
            softWrap: false,
          ),
        ),
    ],
    onChanged: (value) {
      if (value != null) _setCapacity(room, value);
    },
  );

  Widget _lightField(
    TextEditingController controller,
    String hint, {
    IconData? icon,
    bool digits = false,
    bool compact = false,
    bool currency = false,
  }) => TextField(
    controller: controller,
    keyboardType: digits ? TextInputType.number : TextInputType.text,
    inputFormatters: digits
        ? [
            FilteringTextInputFormatter.digitsOnly,
            LengthLimitingTextInputFormatter(7),
          ]
        : [LengthLimitingTextInputFormatter(40)],
    style: GoogleFonts.dmSans(
      color: const Color(0xFF183636),
      fontWeight: FontWeight.w700,
      fontSize: compact ? 10 : 11,
    ),
    decoration: _lightDecoration(
      hint,
      icon,
      compact: compact,
      currency: currency,
    ),
  );

  InputDecoration _lightDecoration(
    String hint,
    IconData? icon, {
    bool compact = false,
    bool currency = false,
  }) => InputDecoration(
    isDense: true,
    hintText: hint,
    prefixIcon: currency
        ? Center(
            widthFactor: 1,
            child: Padding(
              padding: const EdgeInsets.only(left: 9, right: 5),
              child: Text(
                'PKR',
                style: GoogleFonts.dmSans(
                  color: AppTheme.teal,
                  fontSize: 8.5,
                  fontWeight: FontWeight.w800,
                ),
              ),
            ),
          )
        : icon == null
        ? null
        : Icon(icon, color: AppTheme.teal, size: compact ? 15 : 18),
    hintStyle: GoogleFonts.dmSans(
      color: const Color(0xFF708482),
      fontSize: compact ? 9 : 10,
    ),
    filled: true,
    fillColor: Colors.white.withOpacity(.95),
    contentPadding: EdgeInsets.symmetric(
      horizontal: compact ? 8 : 10,
      vertical: compact ? 9 : 12,
    ),
    border: OutlineInputBorder(
      borderRadius: BorderRadius.circular(11),
      borderSide: BorderSide.none,
    ),
    enabledBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(11),
      borderSide: BorderSide(color: Colors.white.withOpacity(.5)),
    ),
    focusedBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(11),
      borderSide: const BorderSide(color: AppTheme.teal, width: 1.4),
    ),
  );

  Widget _sectionTitle(IconData icon, String title) => Row(
    children: [
      Icon(icon, color: AppTheme.tealMint, size: 17),
      const SizedBox(width: 7),
      Text(
        title,
        style: GoogleFonts.dmSans(
          color: Colors.white,
          fontSize: 13,
          fontWeight: FontWeight.w800,
        ),
      ),
    ],
  );
  Widget _pill(String text) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
    decoration: BoxDecoration(
      color: AppTheme.tealMint.withOpacity(.13),
      borderRadius: BorderRadius.circular(15),
      border: Border.all(color: AppTheme.tealMint.withOpacity(.22)),
    ),
    child: Text(
      text,
      style: GoogleFonts.dmSans(
        color: AppTheme.tealMint,
        fontSize: 9,
        fontWeight: FontWeight.w800,
      ),
    ),
  );
  Widget _roundIcon(IconData icon, void Function() onTap) => InkWell(
    onTap: onTap,
    borderRadius: BorderRadius.circular(50),
    child: Container(
      width: 42,
      height: 42,
      decoration: BoxDecoration(
        color: const Color(0xFF02383B).withOpacity(.55),
        shape: BoxShape.circle,
        border: Border.all(color: Colors.white.withOpacity(.22)),
      ),
      child: Icon(icon, color: Colors.white, size: 20),
    ),
  );
  Widget _errorBox() => Container(
    width: double.infinity,
    margin: const EdgeInsets.only(bottom: 10),
    padding: const EdgeInsets.all(10),
    decoration: BoxDecoration(
      color: const Color(0xFF7F2525).withOpacity(.42),
      borderRadius: BorderRadius.circular(12),
    ),
    child: Text(
      _error!,
      style: GoogleFonts.dmSans(color: const Color(0xFFFFB5B5), fontSize: 10.5),
    ),
  );
}

class _FloorDraft {
  final TextEditingController labelController;
  final List<_RoomDraft> rooms;
  _FloorDraft({required this.labelController, required this.rooms});
  factory _FloorDraft.create(int number) => _FloorDraft(
    labelController: TextEditingController(text: 'Floor $number'),
    rooms: [_RoomDraft.create(1)],
  );
  void dispose() {
    labelController.dispose();
    for (final room in rooms) {
      room.dispose();
    }
  }
}

class _RoomDraft {
  final TextEditingController numberController;
  final TextEditingController rentController;
  int capacity;
  final List<TextEditingController> bedControllers;
  _RoomDraft({
    required this.numberController,
    required this.rentController,
    required this.capacity,
    required this.bedControllers,
  });
  factory _RoomDraft.create(int number) => _RoomDraft(
    numberController: TextEditingController(text: '$number'),
    rentController: TextEditingController(),
    capacity: 1,
    bedControllers: [TextEditingController(text: '1')],
  );
  void dispose() {
    numberController.dispose();
    rentController.dispose();
    for (final controller in bedControllers) {
      controller.dispose();
    }
  }
}
